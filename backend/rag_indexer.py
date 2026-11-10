"""
=============================================================================
rag_indexer.py - Build the patient-scoped knowledge base
=============================================================================

This module connects the existing document loader to the RAG stack:

    processed file dict -> stable document record -> chunks -> embeddings
                        -> ChromaDB

Key behaviours:
    * A document ID is derived from the patient ID and filename, so
      re-uploading a corrected version updates the same document.
    * A content hash detects unchanged documents, so they are not embedded
      again on every Streamlit rerun.
    * Patient IDs are random, non-identifying case IDs.  Patient names are
      never used as identifiers.
    * The same uploaded documents feed both the structured extraction
      pipeline and the RAG pipeline.
=============================================================================
"""

import hashlib
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional

from backend.chunker import chunk_document
from backend.config import DEFAULT_CONFIG
from backend.embedding_service import EmbeddingServiceError, get_embedding_service
from backend.hybrid_retriever import HybridRetriever
from backend.rag_engine import RAGEngine
from backend.reranker import Reranker
from backend.retriever import SemanticRetriever
from backend.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


_DATE_PATTERNS = [
    # YYYY-MM-DD
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
    # DD-MM-YYYY or DD/MM/YYYY
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"),
    # DD-Mon-YYYY
    re.compile(r"\b(\d{1,2})[-\s]([A-Za-z]{3,9})[-\s](\d{4})\b"),
    # Month DD, YYYY
    re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"),
]

_DATE_LABEL_PATTERN = re.compile(
    r"\b(?:date|exam date|report date|date of admission|date of discharge|"
    r"date collected|collection date|visit date)\b\s*[:\-]?\s*",
    re.IGNORECASE,
)

_MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def generate_patient_id() -> str:
    """Create a random, non-identifying case ID."""
    return f"case_{uuid.uuid4().hex[:12]}"


def document_content_hash(text: str) -> str:
    """Return the SHA-256 hash used to detect unchanged document text."""
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def index_signature(documents: Iterable[Dict]) -> str:
    """
    Return a short signature of a document set.

    The signature covers each document ID and content hash, so it changes
    whenever a document is added, removed, or edited.  The UI uses it to
    decide whether a cached indexing result is still valid.
    """
    parts = sorted(
        f"{document.get('document_id', '')}:{document.get('content_hash', '')}"
        for document in documents or []
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def safe_filename(filename: str) -> str:
    """
    Reduce a user-supplied filename to a safe display value.

    We keep the original name for citations but strip directory components
    and control characters so it cannot be used for path traversal.
    """
    if not filename:
        return "unknown"
    name = Path(str(filename)).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name or "unknown"


def _iso_date(year, month, day) -> Optional[str]:
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except (TypeError, ValueError):
        return None


def extract_document_date(text: str) -> str:
    """
    Extract a document date and normalise it to ISO where possible.

    Only explicit document/visit/exam date labels are considered; dates of
    birth and treatment dates inside prose are ignored.  If no date can be
    parsed, an empty string is returned rather than a guessed value.
    """
    if not text:
        return ""

    for label_match in _DATE_LABEL_PATTERN.finditer(text):
        window = text[label_match.end(): label_match.end() + 40]
        for pattern in _DATE_PATTERNS:
            match = pattern.search(window)
            if not match:
                continue
            groups = match.groups()
            if len(groups) != 3:
                continue
            first, second, third = groups

            # YYYY-MM-DD
            if pattern is _DATE_PATTERNS[0]:
                iso = _iso_date(first, second, third)
            # DD-MM-YYYY / DD/MM/YYYY
            elif pattern is _DATE_PATTERNS[1]:
                iso = _iso_date(third, second, first)
            # DD-Mon-YYYY
            elif pattern is _DATE_PATTERNS[2]:
                month = _MONTHS.get(second.lower())
                iso = _iso_date(third, month, first) if month else None
            # Month DD, YYYY
            else:
                month = _MONTHS.get(first.lower())
                iso = _iso_date(third, month, second) if month else None

            if iso:
                return iso
            return match.group(0).strip()
    return ""


def build_document_record(
    processed_file: Dict,
    patient_id: str,
    upload_timestamp: Optional[str] = None,
) -> Dict:
    """
    Convert a processed file result into a normalised RAG document record.
    """
    if not isinstance(processed_file, dict):
        raise ValueError("processed_file must be a dictionary.")

    patient_id = str(patient_id or "").strip()
    if not patient_id:
        raise ValueError("A patient_id is required when building a document record.")

    filename = safe_filename(processed_file.get("filename", "unknown"))
    text = processed_file.get("text", "") or ""
    content_hash = document_content_hash(text)
    # Stable per patient + filename so a corrected re-upload replaces the
    # previous version instead of creating a duplicate document.
    document_id = hashlib.sha256(f"{patient_id}|{filename}".encode("utf-8")).hexdigest()[:16]

    pages = processed_file.get("pages") or []
    if not pages and text:
        pages = [{"page_number": None, "text": text}]

    if upload_timestamp is None:
        upload_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return {
        "patient_id": patient_id,
        "document_id": document_id,
        "filename": filename,
        "file_type": processed_file.get("file_type", ""),
        "text": text,
        "pages": pages,
        "content_hash": content_hash,
        "upload_timestamp": upload_timestamp,
        "document_date": extract_document_date(text),
    }


def index_documents(
    documents: Iterable[Dict],
    embedding_service,
    vector_store: VectorStore,
    config=None,
    force: bool = False,
) -> Dict:
    """
    Chunk, embed and store documents for one patient.

    Unchanged documents are skipped by comparing their content hash with the
    metadata already stored in ChromaDB, so re-running this on every question
    does not re-embed anything.

    Returns:
        dict with success, indexed, updated, skipped, chunks_indexed,
        errors, warnings, and documents (per-document status).
    """
    config = config or DEFAULT_CONFIG
    result = {
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
        document_id = document.get("document_id", "")
        try:
            status, warning = _index_single_document(
                document, embedding_service, vector_store, config, force
            )
        except (EmbeddingServiceError, VectorStoreError) as error:
            logger.error("Indexing failed for document %s: %s", document_id, error)
            status = _error_status(document, error)
            result["errors"].append(f"{document.get('filename', 'unknown')}: {error}")
            result["success"] = False
        except Exception as error:  # defensive: one bad file must not stop the rest
            logger.exception("Unexpected indexing error for document %s", document_id)
            status = _error_status(document, error)
            result["errors"].append(f"{document.get('filename', 'unknown')}: {error}")
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
) -> tuple:
    """
    Index one document and return ``(status, warning)``.

    ``status["status"]`` is one of: indexed, updated, unchanged, empty.
    """
    filename = document.get("filename", "unknown")
    patient_id = document.get("patient_id", "")
    document_id = document.get("document_id", "")
    content_hash = document.get("content_hash", "")
    status = {"filename": filename, "document_id": document_id, "status": "pending"}

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


def _tally_document(result: Dict, status: Dict, warning) -> None:
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
    documents: Iterable[Dict],
    components: Dict,
    force: bool = False,
) -> Dict:
    """
    Index documents through an already-built RAG component bundle.

    This is the backend entry point used by the Streamlit UI.  It also
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

    patient_id = documents[0].get("patient_id", "")
    if patient_id:
        components["hybrid_retriever"].invalidate(patient_id)
    return result


def build_rag_components(config=None, embedding_service=None, vector_store=None):
    """
    Build the retrieval and generation objects used by the Streamlit UI.

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
