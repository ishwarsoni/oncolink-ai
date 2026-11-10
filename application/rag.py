"""Evidence retrieval and RAG workflow.

The service owns the lifecycle of the RAG stack for one case:

    build components lazily -> index canonical documents once -> answer

All heavy objects (embedding model, reranker, ChromaDB client) are created on
first use.  Re-indexing is content-hash based, so asking a second question does
not re-embed unchanged documents.  Patient scoping is passed explicitly to
every storage call and is re-validated by ``infrastructure.vector_store``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from backend.rag_indexer import (
    build_rag_components,
    ensure_documents_indexed,
    index_signature,
    remove_patient_index,
)
from domain.documents import ClinicalDocument
from infrastructure.config import RAGConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexStatus:
    """Read-only view of what is currently indexed for one patient."""

    patient_id: str
    documents: tuple[dict[str, Any], ...] = ()
    chunk_count: int = 0
    error: str = ""

    @property
    def success(self) -> bool:
        return not self.error

    @property
    def has_documents(self) -> bool:
        return bool(self.documents)


def _index_failure(message: str, warnings: Iterable[str] = ()) -> dict[str, Any]:
    """Return the answer-shaped failure used when indexing cannot run."""
    return {
        "success": False,
        "answer": "",
        "sources": [],
        "retrieved_evidence": [],
        "unresolved_contradictions": [],
        "insufficient_evidence": True,
        "warnings": list(warnings),
        "error": message,
    }


def _empty_index_result(errors: Iterable[str], warnings: Iterable[str] = ()) -> dict[str, Any]:
    """Return the index result shape used for empty or invalid document sets."""
    return {
        "success": False,
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "chunks_indexed": 0,
        "errors": list(errors),
        "warnings": list(warnings),
        "documents": [],
    }


class RAGService:
    """Lazy, per-case facade over indexing, retrieval and answer generation."""

    def __init__(self, config: RAGConfig | None = None, components: dict | None = None):
        self._config = config
        self._components = components
        self._init_error = ""
        self._documents: tuple[ClinicalDocument, ...] = ()
        self._index_signature: str | None = None
        self._index_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Components
    # ------------------------------------------------------------------
    @property
    def init_error(self) -> str:
        """Message from the last failed component initialisation, if any."""
        return self._init_error

    @property
    def components(self) -> dict | None:
        """Return the RAG component bundle, building it on first use.

        Returns ``None`` instead of raising so the UI can keep the structured
        extraction capability working when ChromaDB or a model is unavailable.
        """
        if self._components is not None:
            return self._components

        try:
            self._components = build_rag_components(config=self._config or RAGConfig())
            self._init_error = ""
        except Exception as error:  # lazy init failure must be recoverable
            logger.error("RAG components could not be initialised: %s", error)
            self._init_error = str(error)
            return None
        return self._components

    # ------------------------------------------------------------------
    # Documents and indexing
    # ------------------------------------------------------------------
    @property
    def documents(self) -> tuple[ClinicalDocument, ...]:
        """Canonical documents currently attached to this service."""
        return self._documents

    def set_documents(self, documents: Sequence[ClinicalDocument]) -> None:
        """Attach the ingested documents and invalidate stale index caches."""
        documents = tuple(documents or ())
        signature = index_signature(documents)
        if signature != index_signature(self._documents):
            self._invalidate_cache()
        self._documents = documents

    def ensure_index(self, force: bool = False) -> dict[str, Any]:
        """Index the attached documents, reusing unchanged embeddings.

        Args:
            force: Re-embed every document even when its content hash matches.

        Returns:
            The index result dictionary.  ``success`` is False when no usable
            documents exist or when indexing failed.
        """
        documents = self._documents
        if not documents:
            return _empty_index_result(
                ["No readable documents are available to index."]
            )

        patient_id = documents[0].patient_id
        if any(document.patient_id != patient_id for document in documents):
            return _empty_index_result(
                ["Refusing to index documents from more than one patient."]
            )

        signature = index_signature(documents)
        if (
            not force
            and self._index_result
            and self._index_signature == signature
            and self._index_result.get("success")
        ):
            components = self.components
            if components is not None:
                try:
                    if components["vector_store"].count(patient_id) > 0:
                        return self._cached_index_result()
                except Exception:
                    # A stale cache must never block a fresh index attempt.
                    pass

        components = self.components
        if components is None:
            return _empty_index_result([self._init_error or "RAG initialisation failed."])

        result = ensure_documents_indexed(documents, components, force=force)
        self._index_result = result
        self._index_signature = signature if result.get("success") else None
        return result

    def index_status(self, patient_id: str) -> IndexStatus:
        """Return the indexed documents and chunk count for one patient."""
        components = self.components
        if components is None:
            return IndexStatus(patient_id=patient_id, error=self._init_error)

        try:
            documents = components["vector_store"].list_documents(patient_id)
            chunk_count = components["vector_store"].count(patient_id)
        except Exception as error:
            logger.error("Could not read the RAG index: %s", error)
            return IndexStatus(patient_id=patient_id, error=str(error))

        return IndexStatus(
            patient_id=patient_id,
            documents=tuple(documents),
            chunk_count=chunk_count,
        )

    def remove_index(self, patient_id: str) -> int:
        """Delete all indexed chunks for one patient and invalidate caches."""
        components = self.components
        if components is None:
            return 0

        deleted = remove_patient_index(components["vector_store"], patient_id)
        components["hybrid_retriever"].invalidate(patient_id)
        self._invalidate_cache()
        return deleted

    # ------------------------------------------------------------------
    # Questions
    # ------------------------------------------------------------------
    def ask(
        self,
        patient_id: str,
        question: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool = False,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Index if needed, retrieve evidence and generate a cited answer."""
        index_result = self.ensure_index()
        if not index_result.get("success"):
            return _index_failure(
                "The RAG index could not be built: "
                + "; ".join(index_result.get("errors", [])),
                warnings=index_result.get("warnings", []),
            )

        components = self.components
        if components is None:
            return _index_failure(self._init_error or "RAG initialisation failed.")

        # Reranking is chosen per question; the model still loads lazily on
        # the first question that actually enables it.
        components["reranker"].enabled = bool(rerank)
        try:
            return components["rag_engine"].answer(
                patient_id=patient_id,
                question=question,
                top_k=top_k,
                mode=mode,
                rerank=rerank,
                score_threshold=score_threshold,
            )
        except Exception as error:  # defensive: a question must not crash the UI
            logger.exception("RAG answer generation failed")
            return _index_failure(f"RAG answer generation failed: {error}")

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------
    def _cached_index_result(self) -> dict[str, Any]:
        """Return a copy of the cached result with current-call counters.

        The cached result describes the call that did the work.  A later call
        skipped everything, so reporting "1 new" again would be misleading.
        """
        result = dict(self._index_result or {})
        result["indexed"] = 0
        result["updated"] = 0
        result["chunks_indexed"] = 0
        result["skipped"] = len(result.get("documents") or [])
        result["cached"] = True
        return result

    def _invalidate_cache(self) -> None:
        """Forget the cached index result and signature."""
        self._index_result = None
        self._index_signature = None
