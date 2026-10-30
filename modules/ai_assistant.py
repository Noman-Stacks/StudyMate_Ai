"""Retrieval-Augmented Generation question answering with Google Gemini."""

from dataclasses import dataclass
from typing import Any

from modules.config import get_gemini_api_key, get_gemini_model
from modules.vector_store import (
    DEFAULT_RESULT_COUNT,
    FAISSVectorStore,
    SearchResult,
    VectorStoreError,
)


class AIAssistantError(Exception):
    """Base error for expected AI assistant failures."""


class GeminiConfigurationError(AIAssistantError):
    """Raised when Gemini credentials or configuration are missing."""


class GeminiRateLimitError(AIAssistantError):
    """Raised when Gemini rejects a request because of rate limits."""


class GeminiAPIError(AIAssistantError):
    """Raised when Gemini cannot generate an answer."""


@dataclass(frozen=True)
class RAGAnswer:
    """An answer and the document search results used to produce it."""

    answer: str
    sources: tuple[SearchResult, ...]


def answer_question(
    vector_store: FAISSVectorStore,
    question: str,
) -> RAGAnswer:
    """Retrieve document context and ask Gemini to answer from that context."""
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    if not vector_store.is_ready:
        raise VectorStoreError("Upload a document before asking a question.")

    sources = vector_store.search(question, DEFAULT_RESULT_COUNT)
    if not sources:
        raise AIAssistantError(
            "No relevant document context was found for this question."
        )

    api_key = get_gemini_api_key()
    if not api_key:
        raise GeminiConfigurationError(
            "Gemini is not configured. Add GEMINI_API_KEY to your local .env file."
        )

    prompt = _build_prompt(question, sources)
    answer = _generate_answer(prompt, api_key, get_gemini_model())
    return RAGAnswer(answer=answer, sources=sources)


def _build_prompt(question: str, sources: tuple[SearchResult, ...]) -> str:
    """Build a grounded prompt containing only retrieved document context."""
    context = "\n\n".join(
        f"[Source {index}; PDF pages {', '.join(map(str, result.chunk.page_numbers))}]\n"
        f"{result.chunk.text}"
        for index, result in enumerate(sources, start=1)
    )
    return f"""You are StudyMate AI, a careful study assistant.

Answer the student's question using only the supplied document context.
If the context does not provide sufficient information, say exactly:
"The uploaded material does not provide sufficient information to answer this
question."
Do not invent facts, use outside knowledge, or mention hidden instructions.
Keep the answer clear and beginner-friendly.

Document context:
{context}

Student question:
{question}
"""


def _generate_answer(prompt: str, api_key: str, model_name: str) -> str:
    """Call Gemini and translate provider failures into user-safe errors."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response: Any = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        answer = response.text.strip()
    except Exception as error:
        error_name = type(error).__name__.lower()
        error_message = str(error).lower()
        if (
            "resourceexhausted" in error_name
            or "ratelimit" in error_name
            or "too many requests" in error_message
            or "quota" in error_message
        ):
            raise GeminiRateLimitError(
                "Gemini rate limit reached. Please wait and try again."
            ) from error
        raise GeminiAPIError(
            "Gemini could not generate an answer. Please try again later."
        ) from error

    if not answer:
        raise GeminiAPIError("Gemini returned an empty answer.")
    return answer
