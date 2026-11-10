"""
=============================================================================
reranker.py - Optional cross-encoder reranking
=============================================================================

A cross-encoder reads the question and a candidate chunk *together*, which
is slower but more precise than comparing two independent embeddings.  We
use it as an optional second stage:

    retrieve a larger candidate set -> rerank -> keep the strongest chunks

Reranking is disabled by default because it downloads another model and
adds latency.  If the model cannot be loaded, the caller receives the
original retrieval order with a warning; the RAG assistant remains usable.
=============================================================================
"""

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_RERANKER_CACHE = {}
_RERANKER_CACHE_LOCK = threading.Lock()


class Reranker:
    """Cross-encoder reranker with a graceful disabled/failed fallback."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        enabled: bool = True,
        batch_size: int = 16,
        model_loader=None,
    ):
        self.model_name = model_name
        self.enabled = bool(enabled)
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._load_failed = False
        self._model_loader = model_loader
        self._infer_lock = threading.Lock()

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_failed:
            return None

        cache_key = self.model_name
        with _RERANKER_CACHE_LOCK:
            if cache_key in _RERANKER_CACHE:
                self._model = _RERANKER_CACHE[cache_key]
                return self._model
            try:
                if self._model_loader is not None:
                    model = self._model_loader(self.model_name)
                else:
                    from sentence_transformers import CrossEncoder

                    model = CrossEncoder(self.model_name)
            except Exception as error:
                logger.warning("Could not load reranker model %s: %s", self.model_name, error)
                self._load_failed = True
                return None
            _RERANKER_CACHE[cache_key] = model
            self._model = model
            return self._model

    def is_available(self) -> bool:
        """Return True if reranking is enabled and the model is loaded/loadable."""
        if not self.enabled:
            return False
        return self._load_model() is not None

    def rerank(
        self,
        query: str,
        results: List[Dict],
        top_k: Optional[int] = None,
    ) -> Dict:
        """
        Score and reorder candidate chunks.

        Returns:
            dict with success, results, error, warnings, reranked (bool),
            score_type.
        """
        response = {
            "success": True,
            "results": list(results or []),
            "error": None,
            "warnings": [],
            "reranked": False,
            "score_type": "retrieval_score",
        }

        if not self.enabled:
            if top_k is not None:
                response["results"] = response["results"][: max(1, int(top_k))]
            return response

        if not results:
            return response

        model = self._load_model()
        if model is None:
            response["warnings"].append(
                "Reranking was requested but the reranker model could not be loaded. "
                "Using the original retrieval order."
            )
            if top_k is not None:
                response["results"] = response["results"][: max(1, int(top_k))]
            return response

        pairs = [[query, item.get("text", "")] for item in results]
        try:
            with self._infer_lock:
                scores = model.predict(pairs, batch_size=self.batch_size)
        except Exception as error:
            logger.error("Reranking failed: %s", error)
            response["warnings"].append(
                f"Reranking failed ({error}). Using the original retrieval order."
            )
            if top_k is not None:
                response["results"] = response["results"][: max(1, int(top_k))]
            return response

        scored = []
        for index, item in enumerate(results):
            entry = dict(item)
            entry["rerank_score"] = float(scores[index])
            entry["score"] = entry["rerank_score"]
            entry["score_type"] = "cross_encoder_score"
            scored.append(entry)

        scored.sort(key=lambda item: item["rerank_score"], reverse=True)
        if top_k is not None:
            scored = scored[: max(1, int(top_k))]

        response["results"] = scored
        response["reranked"] = True
        response["score_type"] = "cross_encoder_score"
        return response
