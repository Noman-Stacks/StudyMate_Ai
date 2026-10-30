"""Focused tests for grounded RAG question answering."""

from unittest import TestCase
from unittest.mock import patch

from modules.ai_assistant import (
    GeminiConfigurationError,
    _build_prompt,
    answer_question,
)
from modules.text_chunker import TextChunk
from modules.vector_store import SearchResult


class FakeVectorStore:
    """Small store double for testing the Gemini boundary."""

    is_ready = True

    def search(self, question: str, result_count: int) -> tuple[SearchResult, ...]:
        chunk = TextChunk(
            text="Photosynthesis uses light to make chemical energy.",
            chunk_index=0,
            page_numbers=(3,),
            document_name="biology.pdf",
        )
        return (SearchResult(chunk=chunk, score=0.91),)


class AIAssistantTest(TestCase):
    """Tests for retrieval context and credential handling."""

    def test_prompt_contains_question_context_and_grounding_rule(self) -> None:
        source = FakeVectorStore().search("question", 5)
        prompt = _build_prompt("How does photosynthesis work?", source)

        self.assertIn("Photosynthesis uses light", prompt)
        self.assertIn("PDF pages 3", prompt)
        self.assertIn("How does photosynthesis work?", prompt)
        self.assertIn("does not provide sufficient information", prompt)

    @patch("modules.ai_assistant._generate_answer", return_value="It uses light.")
    @patch("modules.ai_assistant.get_gemini_api_key", return_value="test-key")
    def test_question_returns_grounded_answer_and_sources(
        self,
        _get_key: object,
        _generate_answer: object,
    ) -> None:
        response = answer_question(
            FakeVectorStore(),
            "How does photosynthesis work?",
        )

        self.assertEqual(response.answer, "It uses light.")
        self.assertEqual(response.sources[0].chunk.page_numbers, (3,))

    @patch("modules.ai_assistant.get_gemini_api_key", return_value=None)
    def test_missing_api_key_is_reported_without_api_call(
        self,
        _get_key: object,
    ) -> None:
        with self.assertRaises(GeminiConfigurationError):
            answer_question(FakeVectorStore(), "How does photosynthesis work?")
