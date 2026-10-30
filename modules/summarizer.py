"""Grounded, section-based PDF summarization with Google Gemini."""

from dataclasses import dataclass
from typing import Any

from modules.config import get_gemini_api_key, get_gemini_model
from modules.pdf_processor import PDFExtractionResult

WORDS_PER_SECTION = 2_500
SUMMARY_LENGTHS = ("Short", "Medium", "Detailed")


class SummarizerError(Exception):
    """Base error for expected summarization failures."""


class SummaryConfigurationError(SummarizerError):
    """Raised when Gemini credentials are not configured."""


class SummaryRateLimitError(SummarizerError):
    """Raised when Gemini rejects a request because of rate limits."""


class SummaryAPIError(SummarizerError):
    """Raised when Gemini cannot generate a summary."""


@dataclass(frozen=True)
class SummarySection:
    """A manageable source section used during summarization."""

    text: str
    page_numbers: tuple[int, ...]


@dataclass(frozen=True)
class SummaryResult:
    """A final summary and the number of source sections processed."""

    summary: str
    section_count: int


def summarize_document(
    extraction: PDFExtractionResult,
    summary_length: str,
) -> SummaryResult:
    """Summarize extracted PDF text using section summaries and synthesis."""
    if summary_length not in SUMMARY_LENGTHS:
        raise ValueError(
            f"Summary length must be one of: {', '.join(SUMMARY_LENGTHS)}."
        )

    sections = _create_sections(extraction)
    if not sections:
        raise SummarizerError("The document contains no extractable text to summarize.")

    api_key = get_gemini_api_key()
    if not api_key:
        raise SummaryConfigurationError(
            "Gemini is not configured. Add GEMINI_API_KEY to your local .env file."
        )

    section_summaries = tuple(
        _generate_text(
            _build_section_prompt(section, summary_length),
            api_key,
            get_gemini_model(),
        )
        for section in sections
    )
    final_summary = _generate_text(
        _build_final_prompt(section_summaries, summary_length),
        api_key,
        get_gemini_model(),
    )
    return SummaryResult(summary=final_summary, section_count=len(sections))


def _create_sections(extraction: PDFExtractionResult) -> tuple[SummarySection, ...]:
    """Group page text into sections without losing source page references."""
    sections: list[SummarySection] = []
    current_words: list[str] = []
    current_pages: set[int] = set()

    for page_number, page_text in enumerate(extraction.page_text, start=1):
        words = page_text.split()
        while words:
            remaining = WORDS_PER_SECTION - len(current_words)
            current_words.extend(words[:remaining])
            del words[:remaining]
            current_pages.add(page_number)
            if len(current_words) == WORDS_PER_SECTION:
                sections.append(
                    SummarySection(
                        " ".join(current_words),
                        tuple(sorted(current_pages)),
                    )
                )
                current_words = []
                current_pages = set()

    if current_words:
        sections.append(
            SummarySection(" ".join(current_words), tuple(sorted(current_pages)))
        )
    return tuple(sections)


def _build_section_prompt(section: SummarySection, summary_length: str) -> str:
    """Build a factual prompt for one source section."""
    target = {
        "Short": "2-4 concise sentences",
        "Medium": "one focused paragraph",
        "Detailed": "several paragraphs covering the major details",
    }[summary_length]
    pages = ", ".join(map(str, section.page_numbers))
    return f"""Summarize this educational PDF section in {target}.
Use only the supplied text. Preserve important scientific terminology,
definitions, relationships, and concepts. Do not invent facts or add outside
knowledge. Write a factual section summary that can be combined with other
section summaries.

Source PDF pages: {pages}
Section text:
{section.text}
"""


def _build_final_prompt(
    section_summaries: tuple[str, ...],
    summary_length: str,
) -> str:
    """Build a synthesis prompt from grounded section summaries."""
    target = {
        "Short": "a concise overview of the main ideas",
        "Medium": "a coherent summary covering the key ideas and concepts",
        "Detailed": "a thorough, well-structured summary preserving important details",
    }[summary_length]
    summaries = "\n\n".join(
        f"Section {index}:\n{summary}"
        for index, summary in enumerate(section_summaries, start=1)
    )
    return f"""Create {target} of the educational PDF from the section summaries
below. Combine repeated ideas smoothly and preserve scientific terminology,
definitions, and important relationships. Use only these summaries and do not
invent facts that are absent from the source material.

Section summaries:
{summaries}
"""


def _generate_text(prompt: str, api_key: str, model_name: str) -> str:
    """Call Gemini and convert provider failures into safe application errors."""
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response: Any = client.models.generate_content(
            model=model_name,
            contents=prompt,
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
            raise SummaryRateLimitError(
                "Gemini rate limit reached while summarizing. Please wait and try again."
            ) from error
        raise SummaryAPIError(
            "Gemini could not generate the summary. Please try again later."
        ) from error

    if not text:
        raise SummaryAPIError("Gemini returned an empty summary.")
    return text
