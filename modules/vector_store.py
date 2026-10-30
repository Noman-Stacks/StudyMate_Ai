"""FAISS-backed semantic search over embedded document chunks."""

from dataclasses import dataclass

import faiss
import numpy as np

from modules.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingError,
    embed_query,
)
from modules.text_chunker import TextChunk

DEFAULT_RESULT_COUNT = 3


class VectorStoreError(Exception):
    """Raised when a vector index cannot be built or searched."""


@dataclass(frozen=True)
class SearchResult:
    """A chunk returned by semantic search with its similarity score."""

    chunk: TextChunk
    score: float


class FAISSVectorStore:
    """In-memory FAISS index whose metadata stays aligned with its vectors."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        self._index: faiss.Index | None = None
        self._chunks: tuple[TextChunk, ...] = ()
        self._model_name = model_name

    @property
    def is_ready(self) -> bool:
        """Return whether the store contains searchable document chunks."""
        return self._index is not None and bool(self._chunks)

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks currently indexed."""
        return len(self._chunks)

    def rebuild(
        self,
        chunks: tuple[TextChunk, ...],
        vectors: np.ndarray,
    ) -> None:
        """Replace the current index with a new set of chunk vectors."""
        if not chunks:
            raise VectorStoreError("Cannot build an index without document chunks.")
        if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
            raise VectorStoreError("Embedding vectors must be a two-dimensional array.")
        if vectors.shape[0] != len(chunks):
            raise VectorStoreError(
                "The number of vectors must match the number of document chunks."
            )
        if vectors.shape[1] == 0:
            raise VectorStoreError("Embedding vectors cannot have zero dimensions.")

        try:
            normalized_vectors = np.ascontiguousarray(
                vectors.astype(np.float32, copy=False)
            )
            index = faiss.IndexFlatIP(normalized_vectors.shape[1])
            index.add(normalized_vectors)
        except Exception as error:
            raise VectorStoreError("Unable to build the FAISS index.") from error

        self._index = index
        self._chunks = chunks

    def search(
        self,
        query: str,
        result_count: int = DEFAULT_RESULT_COUNT,
    ) -> tuple[SearchResult, ...]:
        """Return the most similar chunks for a natural-language query."""
        if not self.is_ready:
            raise VectorStoreError("Upload a document before searching.")
        if result_count <= 0:
            raise ValueError("result_count must be greater than zero.")

        try:
            query_vector = embed_query(query, self._model_name)
            limit = min(result_count, len(self._chunks))
            scores, positions = self._index.search(
                query_vector.astype(np.float32),
                limit,
            )
        except EmbeddingError:
            raise
        except Exception as error:
            raise VectorStoreError("Unable to search the FAISS index.") from error

        return tuple(
            SearchResult(self._chunks[position], float(score))
            for score, position in zip(scores[0], positions[0])
            if position >= 0
        )
