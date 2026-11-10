"""Compatibility import path for ChromaDB storage.

The canonical implementation lives in :mod:`infrastructure.vector_store`.
"""

from infrastructure.vector_store import (
    VectorStore,
    VectorStoreError,
    get_vector_store,
)

__all__ = ["VectorStore", "VectorStoreError", "get_vector_store"]
