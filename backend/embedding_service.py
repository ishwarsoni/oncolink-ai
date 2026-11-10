"""Compatibility import path for the embedding service.

The canonical implementation lives in :mod:`infrastructure.embeddings`.
"""

from infrastructure.embeddings import (
    EmbeddingService,
    EmbeddingServiceError,
    clear_embedding_cache,
    get_embedding_service,
)

__all__ = [
    "EmbeddingService",
    "EmbeddingServiceError",
    "clear_embedding_cache",
    "get_embedding_service",
]
