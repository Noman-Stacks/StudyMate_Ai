"""PDF validation and page-by-page text extraction with PyMuPDF."""

from dataclasses import dataclass

import fitz

MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


class PDFProcessingError(Exception):
    """Base error for expected PDF processing failures."""


class CorruptedPDFError(PDFProcessingError):
    """Raised when the uploaded file is not a readable PDF."""


class PasswordProtectedPDFError(PDFProcessingError):
    """Raised when a PDF requires a password to open."""


@dataclass(frozen=True)
class PDFExtractionResult:
    """Text and metadata extracted from a PDF document."""

    page_count: int
    page_text: tuple[str, ...]

    @property
    def text(self) -> str:
        """Return all page text in document order."""
        return "\n\n".join(page for page in self.page_text if page.strip())

    @property
    def has_extractable_text(self) -> bool:
        """Return whether at least one page contains searchable text."""
        return bool(self.text.strip())


def extract_pdf_text(file_bytes: bytes) -> PDFExtractionResult:
    """Extract text from each page of a PDF byte stream.

    Raises:
        ValueError: If the file is empty or exceeds the upload limit.
        CorruptedPDFError: If PyMuPDF cannot read the PDF.
        PasswordProtectedPDFError: If the PDF requires a password.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"The PDF exceeds the {MAX_FILE_SIZE_MB} MB upload limit."
        )

    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except fitz.FileDataError as error:
        raise CorruptedPDFError(
            "The file could not be read as a valid PDF."
        ) from error

    try:
        if document.needs_pass:
            raise PasswordProtectedPDFError(
                "This PDF is password-protected. Please upload an unlocked PDF."
            )

        page_text = tuple(page.get_text("text") for page in document)
        return PDFExtractionResult(
            page_count=document.page_count,
            page_text=page_text,
        )
    except PasswordProtectedPDFError:
        raise
    except RuntimeError as error:
        raise CorruptedPDFError(
            "The PDF could not be processed. It may be corrupted or invalid."
        ) from error
    finally:
        document.close()
