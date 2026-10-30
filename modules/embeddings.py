"""Sentence Transformer embeddings for document chunks."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from modules.text_chunker import TextChunk

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class EmbeddingError(Exception):
    """Raised when document chunks cannot be embedded."""


@dataclass(frozen=True)
class EmbeddedChunks:
    """Numerical vectors aligned with their source chunks."""

    chunks: tuple[TextChunk, ...]
    vectors: np.ndarray
    model_name: str


@lru_cache(maxsize=1)
def _load_model(model_name: str) -> Any:
    """Load and cache a Sentence Transformer model for the process."""
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)
    except Exception as error:
        raise EmbeddingError(
            f"Unable to load the embedding model '{model_name}'. "
            "Check that sentence-transformers is installed and the model "
            "can be downloaded."
        ) from error


def embed_chunks(
    chunks: tuple[TextChunk, ...],
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddedChunks:
    """Convert text chunks into normalized numerical embedding vectors."""
    if not chunks:
        raise EmbeddingError("No text chunks are available for embedding.")

    try:
        model = _load_model(model_name)
        vectors = model.encode(
            [chunk.text for chunk in chunks],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError(
            "Unable to generate embeddings for the extracted text."
        ) from error

    if not isinstance(vectors, np.ndarray) or vectors.ndim != 2:
        raise EmbeddingError("The embedding model returned an invalid vector shape.")
    if vectors.shape[0] != len(chunks):
        raise EmbeddingError(
            "The number of generated vectors does not match the number of chunks."
        )

    return EmbeddedChunks(
        chunks=chunks,
        vectors=vectors,
        model_name=model_name,
    )


def embed_query(
    query: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    """Convert one search query into a normalized embedding vector."""
    if not query.strip():
        raise EmbeddingError("Search query cannot be empty.")

    try:
        model = _load_model(model_name)
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except EmbeddingError:
        raise
    except Exception as error:
        raise EmbeddingError("Unable to generate an embedding for the query.") from error

    if not isinstance(vector, np.ndarray) or vector.ndim != 2 or vector.shape[0] != 1:
        raise EmbeddingError("The query embedding has an invalid vector shape.")
    return vector

