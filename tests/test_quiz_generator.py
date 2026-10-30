"""Focused tests for grounded, validated quiz generation."""

import json
from unittest import TestCase
from unittest.mock import patch

from modules.pdf_processor import PDFExtractionResult
from modules.quiz_generator import (
    QuizValidationError,
    _validate_quiz,
    generate_quiz,
)


def _quiz_payload(count: int = 5) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": f"Question {index}",
                    "options": ["A", "B", "C", "D"],
                    "correct_index": 1,
                    "explanation": "The document supports option B.",
                }
                for index in range(count)
            ]
        }
    )


class QuizGeneratorTest(TestCase):
    """Tests for JSON validation and Gemini orchestration."""

    def test_valid_json_is_converted_to_quiz(self) -> None:
        quiz = _validate_quiz(_quiz_payload(), 5)

        self.assertEqual(len(quiz.questions), 5)
        self.assertEqual(quiz.questions[0].correct_index, 1)
        self.assertEqual(len(quiz.questions[0].options), 4)

    def test_invalid_option_count_is_rejected(self) -> None:
        payload = json.loads(_quiz_payload())
        payload["questions"][0]["options"] = ["A", "B"]

        with self.assertRaises(QuizValidationError):
            _validate_quiz(json.dumps(payload), 5)

    @patch("modules.quiz_generator.get_gemini_api_key", return_value="test-key")
    @patch(
        "modules.quiz_generator._generate_json",
        return_value=_quiz_payload(5),
    )
    def test_generation_uses_document_text(
        self,
        _generate_json: object,
        _get_key: object,
    ) -> None:
        extraction = PDFExtractionResult(1, ("Photosynthesis uses light energy.",))

        quiz = generate_quiz(extraction, 5)

        self.assertEqual(len(quiz.questions), 5)
