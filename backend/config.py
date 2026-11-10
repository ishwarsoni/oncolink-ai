"""Compatibility import path for RAG configuration.

The canonical implementation lives in :mod:`infrastructure.config`.
"""

from infrastructure.config import DEFAULT_CONFIG, PROJECT_ROOT, RAGConfig

__all__ = ["DEFAULT_CONFIG", "PROJECT_ROOT", "RAGConfig"]
