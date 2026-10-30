"""Focused tests for safe PDF input handling."""

from unittest import TestCase

import fitz

from modules.pdf_processor import (
    CorruptedPDFError,
    MAX_FILE_SIZE_BYTES,
    extract_pdf_text,
)


class PDFProcessorTest(TestCase):
    """Tests for PDF validation and page-wise extraction."""

    def test_extracts_text_and_page_count(self) -> None:
        document = fitz.open()
        document.new_page().insert_text((72, 72), "First page")
        document.new_page().insert_text((72, 72), "Second page")

        result = extract_pdf_text(document.tobytes())

        self.assertEqual(result.page_count, 2)
        self.assertEqual(len(result.page_text), 2)
        self.assertIn("First page", result.page_text[0])

    def test_rejects_corrupted_pdf(self) -> None:
        with self.assertRaises(CorruptedPDFError):
            extract_pdf_text(b"not a PDF")

    def test_rejects_oversized_input_before_parsing(self) -> None:
        with self.assertRaises(ValueError):
            extract_pdf_text(b"x" * (MAX_FILE_SIZE_BYTES + 1))
