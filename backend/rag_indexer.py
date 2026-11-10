"""RAG indexing: canonical documents -> chunks -> embeddings -> ChromaDB.

This module owns the storage-side workflow only.  Ingestion, status tracking
and UI orchestration live in the application layer; retrieval and generation
live in ``backend.retriever`` / ``backend.rag_engine``.

Key behaviours:
    * Document IDs are derived from patient ID and filename, so a corrected
      re-upload updates the same document instead of duplicating it.
    * Content hashes detect unchanged documents, so Streamlit reruns and
      repeated questions never re-embed unchanged text.
    * Changed documents have their previous chunks deleted before the new
      version is written, so stale evidence can never be retrieved.
    * Every operation is scoped to one patient ID; the vector store validates
      that scope again on every row.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Mapping, Optional

from backend.chunker import chunk_document
from backend.hybrid_retriever import HybridRetriever
from backend.rag_engine import RAGEngine
from backend.retriever import SemanticRetriever
from domain.documents import (
    ClinicalDocument,
    document_content_hash,
    extract_document_date,
    generate_patient_id,
    index_signature,
    safe_filename,
)
from infrastructure.config import DEFAULT_CONFIG
from infrastructure.embeddings import EmbeddingServiceError, get_embedding_service
from infrastructure.reranker import Reranker
from infrastructure.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)

__all__ = [
    "build_document_record",
    "build_rag_components",
    "document_content_hash",
    "ensure_documents_indexed",
    "extract_document_date",
    "generate_patient_id",
    "index_documents",
    "index_signature",
    "remove_patient_index",
    "safe_filename",
]


def _as_record(document: Any) -> Dict[str, Any]:
    """Project a canonical document (or legacy mapping) to the chunker shape."""
    if isinstance(document, ClinicalDocument):
        return document.as_index_record()
    if isinstance(document, Mapping):
        return dict(document)
    raise TypeError("Documents must be ClinicalDocument instances or mappings.")


def _field(document: Any, name: str, default: Any = "") -> Any:
    """Read a field from a canonical document or a legacy mapping."""
    if isinstance(document, ClinicalDocument):
        return getattr(document, name, default)
    return document.get(name, default)


def build_document_record(
    processed_file: Dict,
    patient_id: str,
    upload_timestamp: Optional[str] = None,
) -> Dict:
    """Convert a loader result into a normalised RAG document record.

    Retained as a public helper for tests and integrations; the application
    pipeline builds :class:`ClinicalDocument` objects directly.
    """
    if not isinstance(processed_file, dict):
        raise ValueError("processed_file must be a dictionary.")
    document = ClinicalDocument.from_processed_file(
        processed_file,
        patient_id,
        upload_timestamp=upload_timestamp,
    )
    return document.as_index_record()


def index_documents(
    documents: Iterable[Any],
    embedding_service,
    vector_store: VectorStore,
    config=None,
    force: bool = False,
) -> Dict:
    """Chunk, embed and store documents for one patient.

    Unchanged documents are skipped by comparing their content hash with the
    metadata already stored in ChromaDB, so re-running this on every question
    does not re-embed anything.

    Returns:
        dict with success, indexed, updated, skipped, chunks_indexed,
        errors, warnings, and per-document statuses.
    """
    config = config or DEFAULT_CONFIG
    result: Dict[str, Any] = {
        "success": True,
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "chunks_indexed": 0,
        "errors": [],
        "warnings": [],
        "documents": [],
    }

    for document in documents or []:
        try:
            record = _as_record(document)
        except TypeError as error:
            result["errors"].append(str(error))
            result["success"] = False
            continue

        document_id = record.get("document_id", "")
        try:
            status, warning = _index_single_document(
                record, embedding_service, vector_store, config, force
            )
        except (EmbeddingServiceError, VectorStoreError) as error:
            logger.error("Indexing failed for document %s: %s", document_id, error)
            status = _error_status(record, error)
            result["errors"].append(f"{record.get('filename', 'unknown')}: {error}")
            result["success"] = False
        except Exception as error:  # defensive: one bad file must not stop the rest
            logger.exception("Unexpected indexing error for document %s", document_id)
            status = _error_status(record, error)
            result["errors"].append(f"{record.get('filename', 'unknown')}: {error}")
            result["success"] = False
        else:
            _tally_document(result, status, warning)

        result["documents"].append(status)

    return result


def _index_single_document(
    document: Dict,
    embedding_service,
    vector_store: VectorStore,
    config,
    force: bool,
) -> tuple[Dict, Optional[str]]:
    """Index one document and return ``(status, warning)``.

    ``status["status"]`` is one of: indexed, updated, unchanged, empty.
    """
    filename = document.get("filename", "unknown")
    patient_id = document.get("patient_id", "")
    document_id = document.get("document_id", "")
    content_hash = document.get("content_hash", "")
    status: Dict[str, Any] = {
        "filename": filename,
        "document_id": document_id,
        "status": "pending",
    }

    if not patient_id or not document_id:
        raise ValueError("Document is missing patient_id or document_id.")

    if not (document.get("text") or "").strip():
        status["status"] = "empty"
        return status, f"Skipped empty document: {filename}"

    existing = vector_store.get_document_metadata(patient_id, document_id)
    if existing and existing.get("content_hash") == content_hash and not force:
        status["status"] = "unchanged"
        return status, None

    if existing:
        # Content changed: remove the old chunks before inserting the new
        # version so stale evidence can never be retrieved.
        vector_store.delete_document(patient_id, document_id)
        status["status"] = "updated"

    chunks = chunk_document(
        document,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    if not chunks:
        status["status"] = "empty"
        return status, f"No chunks produced for: {filename}"

    embeddings = embedding_service.embed_documents(
        [chunk["text"] for chunk in chunks],
        batch_size=config.embedding_batch_size,
    )
    stored = vector_store.upsert_chunks(patient_id, chunks, embeddings)

    status["status"] = "updated" if existing else "indexed"
    status["chunks"] = stored
    return status, None


def _error_status(document: Dict, error: Exception) -> Dict:
    """Build the per-document status entry for a failed index attempt."""
    return {
        "filename": document.get("filename", "unknown"),
        "document_id": document.get("document_id", ""),
        "status": "error",
        "error": str(error),
    }


def _tally_document(result: Dict, status: Dict, warning: Optional[str]) -> None:
    """Update the aggregate counters for one successfully processed document."""
    if warning:
        result["warnings"].append(warning)

    state = status["status"]
    if state == "indexed":
        result["indexed"] += 1
        result["chunks_indexed"] += status.get("chunks", 0)
    elif state == "updated":
        result["updated"] += 1
        result["chunks_indexed"] += status.get("chunks", 0)
    elif state in ("empty", "unchanged"):
        result["skipped"] += 1


def ensure_documents_indexed(
    documents: Iterable[Any],
    components: Dict,
    force: bool = False,
) -> Dict:
    """Index documents through an already-built RAG component bundle.

    This is the backend entry point used by the application service.  It also
    invalidates the patient's cached BM25 index because the chunk set changed.
    """
    documents = list(documents or [])
    if not documents:
        return {
            "success": False,
            "indexed": 0,
            "updated": 0,
            "skipped": 0,
            "chunks_indexed": 0,
            "errors": ["No readable documents are available to index."],
            "warnings": [],
            "documents": [],
        }

    result = index_documents(
        documents,
        components["embedding_service"],
        components["vector_store"],
        components["config"],
        force=force,
    )

    patient_id = _field(documents[0], "patient_id", "")
    if patient_id:
        components["hybrid_retriever"].invalidate(patient_id)
    return result


def build_rag_components(config=None, embedding_service=None, vector_store=None):
    """Build the retrieval and generation objects used by the application.

    Models are lazy: creating these objects does not download anything until
    the first embedding or rerank call.
    """
    config = config or DEFAULT_CONFIG

    if embedding_service is None:
        embedding_service = get_embedding_service(
            model_name=config.embedding_model,
            device=config.embedding_device,
            query_prefix=config.query_prefix,
            batch_size=config.embedding_batch_size,
        )

    if vector_store is None:
        vector_store = VectorStore(
            persist_dir=config.chroma_dir,
            collection_name=config.collection_name,
        )

    semantic_retriever = SemanticRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        config=config,
    )
    hybrid_retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        vector_store=vector_store,
        config=config,
    )
    reranker = Reranker(
        model_name=config.reranker_model,
        enabled=config.rerank_enabled,
    )
    rag_engine = RAGEngine(
        retriever=hybrid_retriever,
        reranker=reranker,
        config=config,
    )

    return {
        "config": config,
        "embedding_service": embedding_service,
        "vector_store": vector_store,
        "semantic_retriever": semantic_retriever,
        "hybrid_retriever": hybrid_retriever,
        "reranker": reranker,
        "rag_engine": rag_engine,
    }


def remove_patient_index(vector_store: VectorStore, patient_id: str) -> int:
    """Delete all indexed chunks for a patient."""
    return vector_store.delete_patient(patient_id)
