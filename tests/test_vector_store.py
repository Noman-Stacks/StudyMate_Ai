"""Focused test for retrieving a relevant paragraph from a PDF."""

from unittest import TestCase
from unittest.mock import patch

import fitz
import numpy as np

from modules.pdf_processor import extract_pdf_text
from modules.text_chunker import create_text_chunks
from modules.vector_store import FAISSVectorStore


class VectorStoreTest(TestCase):
    """Tests for retrieving relevant paragraphs from PDF content."""

    def test_question_retrieves_relevant_pdf_chunk(self) -> None:
        """A query returns the chunk associated with matching PDF content."""
        document = fitz.open()
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Photosynthesis converts light energy into chemical energy in plants.",
        )
        page.insert_text(
            (72, 120),
            "The water cycle describes evaporation, condensation, and precipitation.",
        )

        extraction = extract_pdf_text(document.tobytes())
        chunks = create_text_chunks(
            extraction,
            "biology.pdf",
            chunk_size=8,
            overlap=2,
        )
        vectors = np.eye(len(chunks), dtype=np.float32)
        store = FAISSVectorStore()
        store.rebuild(chunks, vectors)

        with patch(
            "modules.vector_store.embed_query",
            return_value=np.array(
                [[1] + [0] * (len(chunks) - 1)],
                dtype=np.float32,
            ),
        ):
            results = store.search("How does photosynthesis work?")

        self.assertTrue(results)
        self.assertIn("Photosynthesis", results[0].chunk.text)
        self.assertEqual(results[0].chunk.page_numbers, (1,))
