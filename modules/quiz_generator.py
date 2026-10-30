"""Grounded multiple-choice quiz generation with Google Gemini."""

from dataclasses import dataclass
import json
from typing import Any

from modules.config import get_gemini_api_key, get_gemini_model
from modules.pdf_processor import PDFExtractionResult

QUESTION_COUNTS = (5, 10, 15)
OPTION_COUNT = 4
MAX_CONTEXT_WORDS = 12_000


class QuizGeneratorError(Exception):
    """Base error for expected quiz-generation failures."""


class QuizConfigurationError(QuizGeneratorError):
    """Raised when Gemini credentials are not configured."""


class QuizRateLimitError(QuizGeneratorError):
    """Raised when Gemini rejects a request because of rate limits."""


class QuizAPIError(QuizGeneratorError):
    """Raised when Gemini cannot generate a quiz."""


class QuizValidationError(QuizGeneratorError):
    """Raised when Gemini returns invalid quiz JSON."""


@dataclass(frozen=True)
class QuizQuestion:
    """One validated multiple-choice question."""

    question: str
    options: tuple[str, str, str, str]
    correct_index: int
    explanation: str


@dataclass(frozen=True)
class Quiz:
    """A validated quiz generated from one document."""

    questions: tuple[QuizQuestion, ...]


def generate_quiz(
    extraction: PDFExtractionResult,
    question_count: int,
) -> Quiz:
    """Generate and validate a document-grounded quiz."""
    if question_count not in QUESTION_COUNTS:
        raise ValueError(
            f"Question count must be one of: {', '.join(map(str, QUESTION_COUNTS))}."
        )

    context = _document_context(extraction)
    if not context:
        raise QuizGeneratorError(
            "The document contains no extractable text for quiz generation."
        )

    api_key = get_gemini_api_key()
    if not api_key:
        raise QuizConfigurationError(
            "Gemini is not configured. Add GEMINI_API_KEY to your local .env file."
        )

    response_text = _generate_json(
        _build_prompt(context, question_count),
        api_key,
        get_gemini_model(),
    )
    return _validate_quiz(response_text, question_count)


def _document_context(extraction: PDFExtractionResult) -> str:
    """Build bounded, page-labeled context from the extracted PDF."""
    words: list[str] = []
    for page_number, page_text in enumerate(extraction.page_text, start=1):
        page_words = page_text.split()
        if not page_words:
            continue
        words.extend([f"[PDF page {page_number}]"] + page_words)
        if len(words) >= MAX_CONTEXT_WORDS:
            break
    return " ".join(words[:MAX_CONTEXT_WORDS])


def _build_prompt(context: str, question_count: int) -> str:
    """Build a strict JSON-only, document-grounded quiz prompt."""
    return f"""Create a multiple-choice quiz from only the supplied educational
PDF context. Generate exactly {question_count} questions.

Every question must:
- test information explicitly supported by the context;
- have exactly four distinct answer options;
- have exactly one correct option;
- use a zero-based integer correct_index from 0 to 3;
- include a short explanation grounded in the context.

Do not use outside knowledge, make assumptions, or ask about information
missing from the context. Preserve important scientific terminology.
Return only valid JSON with this exact shape:
{{"questions":[{{"question":"...", "options":["...", "...", "...", "..."],
"correct_index":0, "explanation":"..."}}]}}

PDF context:
{context}
"""


def _generate_json(prompt: str, api_key: str, model_name: str) -> str:
    """Call Gemini with JSON response configuration and safe error mapping."""
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response: Any = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )
        text = response.text.strip()
    except Exception as error:
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        if (
            "resourceexhausted" in error_name
            or "ratelimit" in error_name
            or "too many requests" in error_message
            or "quota" in error_message
        ):
            raise QuizRateLimitError(
                "Gemini rate limit reached while generating the quiz. "
                "Please wait and try again."
            ) from error
        raise QuizAPIError(
            "Gemini could not generate the quiz. Please try again later."
        ) from error

    if not text:
        raise QuizAPIError("Gemini returned an empty quiz response.")
    return text


def _validate_quiz(response_text: str, question_count: int) -> Quiz:
    """Parse and validate Gemini's structured quiz response."""
    try:
        payload = json.loads(_remove_json_fences(response_text))
    except json.JSONDecodeError as error:
        raise QuizValidationError(
            "Gemini returned malformed quiz data. Please generate the quiz again."
        ) from error

    questions = payload.get("questions") if isinstance(payload, dict) else None
    if not isinstance(questions, list) or len(questions) != question_count:
        raise QuizValidationError(
            f"Gemini returned an invalid question count; expected {question_count}."
        )

    validated: list[QuizQuestion] = []
    for item in questions:
        if not isinstance(item, dict):
            raise QuizValidationError("A generated question has an invalid format.")
        question = item.get("question")
        options = item.get("options")
        correct_index = item.get("correct_index")
        explanation = item.get("explanation")
        if (
            not isinstance(question, str)
            or not question.strip()
            or not isinstance(options, list)
            or len(options) != OPTION_COUNT
            or any(not isinstance(option, str) or not option.strip() for option in options)
            or len({option.strip().casefold() for option in options}) != OPTION_COUNT
            or type(correct_index) is not int
            or not 0 <= correct_index < OPTION_COUNT
            or not isinstance(explanation, str)
            or not explanation.strip()
        ):
            raise QuizValidationError(
                "Gemini returned a question with invalid options or answer data."
            )
        validated.append(
            QuizQuestion(
                question=question.strip(),
                options=tuple(option.strip() for option in options),
                correct_index=correct_index,
                explanation=explanation.strip(),
            )
        )
    return Quiz(questions=tuple(validated))


def _remove_json_fences(response_text: str) -> str:
    """Accept JSON wrapped in markdown fences without weakening validation."""
    cleaned = response_text.strip()
    if cleaned.startswith("```") and cleaned.endswith("```"):
        lines = cleaned.splitlines()
        return "\n".join(lines[1:-1]).removeprefix("json").strip()
    return cleaned
