"""Compatibility import path for the cross-encoder reranker.

The canonical implementation lives in :mod:`infrastructure.reranker`.
"""

from infrastructure.reranker import Reranker

__all__ = ["Reranker"]
