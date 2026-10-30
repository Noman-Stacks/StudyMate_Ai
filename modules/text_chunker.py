"""Word-based document chunking with page and document metadata."""

from dataclasses import dataclass

from modules.pdf_processor import PDFExtractionResult

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


class ChunkingError(Exception):
    """Raised when document text cannot be chunked."""


@dataclass(frozen=True)
class TextChunk:
    """A document chunk and the source metadata needed to trace it."""

    text: str
    chunk_index: int
    page_numbers: tuple[int, ...]
    document_name: str


def create_text_chunks(
    extraction: PDFExtractionResult,
    document_name: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> tuple[TextChunk, ...]:
    """Split extracted PDF text into overlapping word chunks.

    Page numbers are one-based and include every page contributing text to a
    chunk. Empty pages are skipped while preserving the original page numbers.
    """
    if not document_name.strip():
        raise ChunkingError("A document name is required.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be at least zero and smaller than chunk_size.")

    words_with_pages: list[tuple[str, int]] = []
    for page_number, page_text in enumerate(extraction.page_text, start=1):
        words_with_pages.extend(
            (word, page_number) for word in page_text.split() if word
        )

    if not words_with_pages:
        return ()

    step = chunk_size - overlap
    chunks: list[TextChunk] = []
    for chunk_index, start in enumerate(range(0, len(words_with_pages), step)):
        chunk_words = words_with_pages[start : start + chunk_size]
        if not chunk_words:
            break
        chunks.append(
            TextChunk(
                text=" ".join(word for word, _ in chunk_words),
                chunk_index=chunk_index,
                page_numbers=tuple(sorted({page for _, page in chunk_words})),
                document_name=document_name,
            )
        )
        if start + chunk_size >= len(words_with_pages):
            break

    return tuple(chunks)
