"""
=============================================================================
hybrid_retriever.py - BM25 + semantic retrieval with Reciprocal Rank Fusion
=============================================================================

Why hybrid retrieval:
    Oncology records are full of exact terms that dense embeddings can blur:
    EGFR, HER2, PD-L1, ALK, KRAS, ECOG, T3 N2 M0, drug names, and lab values.
    BM25 catches exact terminology; sentence embeddings catch paraphrases.
    Reciprocal Rank Fusion (RRF) combines both rankings without assuming the
    two score scales are comparable.

BM25 index lifecycle:
    The index is built lazily from the patient's chunks and cached.  The
    cache key includes a signature of the patient's indexed documents, so a
    re-index invalidates it automatically.
=============================================================================
"""

import logging
import re
from typing import Dict, Iterable, List, Optional, Tuple

from infrastructure.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


# Keep clinical terms such as PD-L1, HER-2, Ki-67, T3 and ECOG intact.
# Hyphens are removed for matching so PD-L1 and PDL1 both become "pdl1".
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]*")


def tokenize(text: str) -> List[str]:
    """Lowercase and normalize text into BM25 tokens."""
    if not text:
        return []
    tokens = []
    for match in _TOKEN_PATTERN.findall(text):
        token = match.lower()
        token = token.replace("-", "").replace("_", "")
        token = token.strip(".")
        if token:
            tokens.append(token)
    return tokens


class HybridRetriever:
    """Combine dense and keyword retrieval for one patient at a time."""

    def __init__(self, semantic_retriever, vector_store, config=None):
        self.semantic_retriever = semantic_retriever
        self.vector_store = vector_store
        self.config = config or DEFAULT_CONFIG
        self._bm25_cache: Dict[Tuple, Dict] = {}

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------
    def _cache_key(self, patient_id: str, document_ids: Optional[Iterable[str]]):
        doc_key = tuple(sorted(str(doc) for doc in (document_ids or [])))
        signature = self.vector_store.patient_signature(patient_id)
        return patient_id, doc_key, signature

    def invalidate(self, patient_id: Optional[str] = None):
        """Drop cached BM25 indexes (all, or one patient's)."""
        if patient_id is None:
            self._bm25_cache.clear()
            return
        for key in list(self._bm25_cache):
            if key[0] == patient_id:
                self._bm25_cache.pop(key, None)

    def _get_bm25(self, patient_id: str, document_ids: Optional[Iterable[str]] = None):
        """Build or reuse the BM25 index for a patient."""
        key = self._cache_key(patient_id, document_ids)
        cached = self._bm25_cache.get(key)
        if cached is not None:
            return cached

        chunks = self.vector_store.get_all_chunks(patient_id, document_ids=document_ids)
        tokenized = [tokenize(chunk.get("text", "")) for chunk in chunks]

        if not chunks:
            entry = {"chunks": [], "tokenized": [], "bm25": None}
            self._bm25_cache[key] = entry
            return entry

        try:
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(tokenized)
        except Exception as error:
            logger.error("Could not build BM25 index: %s", error)
            bm25 = None

        entry = {"chunks": chunks, "tokenized": tokenized, "bm25": bm25}
        # Keep the cache bounded; a demo app does not need unlimited history.
        if len(self._bm25_cache) > 32:
            self._bm25_cache.pop(next(iter(self._bm25_cache)))
        self._bm25_cache[key] = entry
        return entry

    def keyword_search(
        self,
        patient_id: str,
        query: str,
        top_k: int = 5,
        document_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        """Return chunks ranked by BM25 keyword relevance."""
        entry = self._get_bm25(patient_id, document_ids)
        bm25 = entry["bm25"]
        chunks = entry["chunks"]
        query_tokens = tokenize(query)

        if bm25 is None or not chunks or not query_tokens:
            return []

        try:
            scores = bm25.get_scores(query_tokens)
        except Exception as error:
            logger.error("BM25 scoring failed: %s", error)
            return []

        ranked = []
        for index, score in enumerate(scores):
            if float(score) <= 0:
                continue
            chunk = dict(chunks[index])
            chunk["keyword_score"] = float(score)
            ranked.append(chunk)

        ranked.sort(key=lambda item: item["keyword_score"], reverse=True)
        return ranked[: max(1, int(top_k))]

    # ------------------------------------------------------------------
    # Public retrieval API
    # ------------------------------------------------------------------
    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: Optional[int] = None,
        mode: str = "hybrid",
        document_ids: Optional[Iterable[str]] = None,
        score_threshold: Optional[float] = None,
        candidate_k: Optional[int] = None,
    ) -> Dict:
        """
        Retrieve evidence using semantic-only or hybrid search.

        mode: "semantic" or "hybrid" (default).
        """
        top_k = max(1, int(top_k or self.config.top_k))
        candidate_k = max(top_k, int(candidate_k or self.config.candidate_k))
        mode = (mode or "hybrid").strip().lower()
        if mode not in {"semantic", "hybrid"}:
            mode = "hybrid"

        result = {
            "success": False,
            "query": query,
            "mode": mode,
            "results": [],
            "error": None,
            "warnings": [],
            "score_type": "cosine_similarity" if mode == "semantic" else "rrf",
        }

        if not isinstance(patient_id, str) or not patient_id.strip():
            result["error"] = "A patient_id is required for retrieval."
            return result
        if not isinstance(query, str) or not query.strip():
            result["error"] = "Please enter a non-empty question."
            return result

        semantic_result = self.semantic_retriever.retrieve(
            patient_id=patient_id,
            query=query,
            top_k=candidate_k,
            document_ids=document_ids,
            score_threshold=score_threshold,
        )

        semantic_items = semantic_result.get("results", [])
        if semantic_result.get("error"):
            result["warnings"].append(semantic_result["error"])

        if mode == "semantic":
            result["results"] = semantic_items[:top_k]
            result["success"] = semantic_result.get("success", False)
            result["error"] = semantic_result.get("error")
            result["warnings"].extend(semantic_result.get("warnings", []))
            return result

        keyword_items = self.keyword_search(
            patient_id=patient_id,
            query=query,
            top_k=candidate_k,
            document_ids=document_ids,
        )

        if not semantic_items and not keyword_items:
            result["success"] = True
            result["warnings"].append("No relevant evidence was found for this question.")
            result["warnings"].extend(semantic_result.get("warnings", []))
            return result

        fused = self._reciprocal_rank_fusion(
            semantic_items=semantic_items,
            keyword_items=keyword_items,
            rrf_k=self.config.rrf_k,
        )
        result["results"] = fused[:top_k]
        result["success"] = True
        result["warnings"].extend(semantic_result.get("warnings", []))
        return result

    @staticmethod
    def _reciprocal_rank_fusion(
        semantic_items: List[Dict],
        keyword_items: List[Dict],
        rrf_k: int = 60,
    ) -> List[Dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        fused: Dict[str, Dict] = {}

        def add_items(items, source):
            for rank, item in enumerate(items, start=1):
                chunk_id = item.get("chunk_id", "")
                if not chunk_id:
                    continue
                if chunk_id not in fused:
                    fused[chunk_id] = dict(item)
                    fused[chunk_id]["rrf_score"] = 0.0
                    fused[chunk_id]["semantic_rank"] = None
                    fused[chunk_id]["keyword_rank"] = None
                    fused[chunk_id]["keyword_score"] = None
                entry = fused[chunk_id]
                entry["rrf_score"] += 1.0 / (rrf_k + rank)
                if source == "semantic":
                    entry["semantic_rank"] = rank
                    entry["similarity"] = item.get("similarity")
                    entry["distance"] = item.get("distance")
                else:
                    entry["keyword_rank"] = rank
                    entry["keyword_score"] = item.get("keyword_score")

        add_items(semantic_items, "semantic")
        add_items(keyword_items, "keyword")

        for entry in fused.values():
            entry["score"] = entry["rrf_score"]
            entry["score_type"] = "rrf"

        return sorted(fused.values(), key=lambda item: item["rrf_score"], reverse=True)
