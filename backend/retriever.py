"""
=============================================================================
retriever.py - Semantic retrieval over ChromaDB
=============================================================================

This is the "dense" half of the retrieval layer.  It embeds the user's
question, asks ChromaDB for the nearest chunks for the active patient, and
returns structured evidence objects.

Score semantics:
    The ChromaDB collection uses cosine space.  Chroma returns a *distance*
    (0 = identical, 2 = opposite).  We convert it to cosine similarity
    (1 - distance) and expose both values.  A raw distance is never labelled
    as a percentage or confidence score.
=============================================================================
"""

import logging
import re
from typing import Dict, Iterable, List, Optional

from backend.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


class SemanticRetriever:
    """Dense vector retrieval scoped to one patient."""

    def __init__(self, embedding_service, vector_store, config=None):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.config = config or DEFAULT_CONFIG

    def retrieve(
        self,
        patient_id: str,
        query: str,
        top_k: Optional[int] = None,
        document_ids: Optional[Iterable[str]] = None,
        score_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Retrieve the most relevant chunks for a question.

        Returns a dict with:
            success (bool), query (str), results (list), error (str|None),
            warnings (list), score_type (str).
        """
        result = {
            "success": False,
            "query": query,
            "results": [],
            "error": None,
            "warnings": [],
            "score_type": "cosine_similarity",
        }

        if not isinstance(patient_id, str) or not patient_id.strip():
            result["error"] = "A patient_id is required for retrieval."
            return result
        if not isinstance(query, str) or not query.strip():
            result["error"] = "Please enter a non-empty question."
            return result

        top_k = max(1, int(top_k or self.config.top_k))
        if score_threshold is None:
            score_threshold = self.config.effective_score_threshold()

        try:
            query_embedding = self.embedding_service.embed_query(query)
        except Exception as error:
            logger.error("Query embedding failed: %s", error)
            result["error"] = f"Could not embed the question: {error}"
            return result

        try:
            raw_results = self.vector_store.query(
                patient_id=patient_id,
                query_embedding=query_embedding,
                top_k=top_k,
                document_ids=document_ids,
            )
        except Exception as error:
            logger.error("Semantic retrieval failed: %s", error)
            result["error"] = f"Vector search failed: {error}"
            return result

        # Deduplicate by chunk ID and by normalized text (defensive against
        # duplicate indexing from older data).
        seen_ids = set()
        seen_text = set()
        filtered: List[Dict] = []

        for item in raw_results:
            chunk_id = item.get("chunk_id", "")
            text_key = _normalise_text(item.get("text", ""))
            if not chunk_id or chunk_id in seen_ids or text_key in seen_text:
                continue
            seen_ids.add(chunk_id)
            seen_text.add(text_key)

            similarity = item.get("similarity")
            if score_threshold is not None and similarity is not None:
                if similarity < score_threshold:
                    continue
            filtered.append(item)

        filtered.sort(key=lambda item: item.get("similarity", -1.0), reverse=True)
        result["results"] = filtered
        result["success"] = True

        if not filtered:
            result["warnings"].append(
                "No sufficiently relevant evidence was found for this question."
            )

        return result
