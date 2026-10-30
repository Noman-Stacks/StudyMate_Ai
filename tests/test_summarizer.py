"""Focused tests for grounded, section-based summarization."""

from unittest import TestCase
from unittest.mock import patch

from modules.pdf_processor import PDFExtractionResult
from modules.summarizer import (
    SummaryConfigurationError,
    _create_sections,
    _build_section_prompt,
    summarize_document,
)


class SummarizerTest(TestCase):
    """Tests for sectioning and Gemini summarization orchestration."""

    def test_long_document_is_split_with_page_references(self) -> None:
        extraction = PDFExtractionResult(
            page_count=2,
            page_text=(" ".join(["alpha"] * 2_500), "beta concept definition"),
        )

        sections = _create_sections(extraction)

        self.assertEqual(len(sections), 2)
        self.assertEqual(sections[0].page_numbers, (1,))
        self.assertEqual(sections[1].page_numbers, (2,))

    def test_summary_combines_section_results(self) -> None:
        extraction = PDFExtractionResult(
            page_count=2,
            page_text=("first concept", "second definition"),
        )
        generated_prompts: list[str] = []

        def fake_generate(prompt: str, api_key: str, model_name: str) -> str:
            generated_prompts.append(prompt)
            return f"summary {len(generated_prompts)}"

        with (
            patch(
                "modules.summarizer.get_gemini_api_key",
                return_value="test-key",
            ),
            patch(
                "modules.summarizer._generate_text",
                side_effect=fake_generate,
            ),
        ):
            result = summarize_document(extraction, "Detailed")

        self.assertEqual(result.section_count, 1)
        self.assertEqual(result.summary, "summary 2")
        self.assertIn("several paragraphs", generated_prompts[0])
        self.assertIn("summary 1", generated_prompts[1])

    @patch("modules.summarizer.get_gemini_api_key", return_value=None)
    def test_missing_api_key_is_reported(self, _get_key: object) -> None:
        extraction = PDFExtractionResult(1, ("document text",))

        with self.assertRaises(SummaryConfigurationError):
            summarize_document(extraction, "Short")

    def test_section_prompt_preserves_source_pages(self) -> None:
        extraction = PDFExtractionResult(1, ("scientific definition",))
        section = _create_sections(extraction)[0]

        prompt = _build_section_prompt(section, "Medium")

        self.assertIn("Source PDF pages: 1", prompt)
        self.assertIn("scientific definition", prompt)
