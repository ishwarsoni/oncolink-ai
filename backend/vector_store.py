"""
=============================================================================
vector_store.py - Persistent ChromaDB storage with patient isolation
=============================================================================

Every chunk stored here carries a non-identifying patient_id.  All read and
write operations require a patient_id and apply it as a metadata filter, so
one patient's documents can never leak into another patient's results.

Why a single collection with metadata filters:
    ChromaDB supports one collection per patient, but creating hundreds of
    collections complicates lifecycle management.  A single collection with
    a mandatory patient_id filter is simpler and is validated again on every
    returned row.  If a row ever has the wrong patient_id it is discarded and
    a warning is logged.

Metadata values:
    ChromaDB only accepts str/int/float/bool metadata.  Missing page numbers
    are stored as -1 and converted back to None by callers.  Patient names
    are never used as identifiers.
=============================================================================
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from backend.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when ChromaDB cannot be initialised or queried."""


def _to_chroma_metadata(chunk: Dict) -> Dict:
    """Convert a chunk dict into ChromaDB-safe metadata."""
    page_number = chunk.get("page_number")
    if page_number is None:
        page_number = -1

    metadata = {
        "patient_id": str(chunk.get("patient_id", "")),
        "document_id": str(chunk.get("document_id", "")),
        "filename": str(chunk.get("filename", "Unknown document")),
        "file_type": str(chunk.get("file_type", "")),
        "page_number": int(page_number),
        "chunk_id": str(chunk.get("chunk_id", "")),
        "chunk_index": int(chunk.get("chunk_index", 0)),
        "section_heading": str(chunk.get("section_heading", "")),
        "upload_timestamp": str(chunk.get("upload_timestamp", "")),
        "document_date": str(chunk.get("document_date", "")),
        "content_hash": str(chunk.get("content_hash", "")),
        "token_count": int(chunk.get("token_count", 0)),
        "char_count": int(chunk.get("char_count", 0)),
    }
    return metadata


def _from_chroma_row(chunk_id: str, document: Optional[str], metadata: Optional[Dict], distance=None) -> Dict:
    """Build a uniform result dict from ChromaDB's parallel result lists."""
    metadata = dict(metadata or {})
    page_number = metadata.get("page_number", -1)
    if page_number is not None and int(page_number) < 0:
        page_number = None

    result = {
        "chunk_id": chunk_id,
        "text": document or "",
        "patient_id": metadata.get("patient_id", ""),
        "document_id": metadata.get("document_id", ""),
        "filename": metadata.get("filename", "Unknown document"),
        "file_type": metadata.get("file_type", ""),
        "page_number": page_number,
        "chunk_index": metadata.get("chunk_index", 0),
        "section_heading": metadata.get("section_heading", ""),
        "upload_timestamp": metadata.get("upload_timestamp", ""),
        "document_date": metadata.get("document_date", ""),
        "content_hash": metadata.get("content_hash", ""),
        "token_count": metadata.get("token_count", 0),
        "char_count": metadata.get("char_count", 0),
        "metadata": metadata,
    }

    if distance is not None:
        distance = float(distance)
        result["distance"] = distance
        # The collection uses cosine space, where distance = 1 - cosine
        # similarity.  Convert it explicitly so callers never treat a raw
        # distance as a similarity percentage.
        result["similarity"] = max(-1.0, min(1.0, 1.0 - distance))
        result["score"] = result["similarity"]

    return result


class VectorStore:
    """Thin, patient-aware wrapper around a persistent ChromaDB collection."""

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: Optional[str] = None,
        client=None,
    ):
        self.persist_dir = str(persist_dir or DEFAULT_CONFIG.chroma_dir)
        self.collection_name = collection_name or DEFAULT_CONFIG.collection_name

        if client is not None:
            self._client = client
        else:
            try:
                import chromadb
                from chromadb.config import Settings

                Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir,
                    settings=Settings(anonymized_telemetry=False),
                )
            except Exception as error:
                logger.error("Could not initialise ChromaDB at %s: %s", self.persist_dir, error)
                raise VectorStoreError(
                    "Could not initialise ChromaDB. Check that the storage directory "
                    f"'{self.persist_dir}' is writable and that chromadb is installed."
                ) from error

        try:
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as error:
            logger.error("Could not open ChromaDB collection %s: %s", self.collection_name, error)
            raise VectorStoreError(
                f"Could not open the ChromaDB collection '{self.collection_name}'. "
                "The database may be corrupted or the collection configuration may "
                "have changed. Move or remove the storage directory and re-index."
            ) from error

        # Read caches keyed by patient_id.  Hybrid retrieval asks for the
        # patient signature on every query, and listing documents/counting
        # chunks scans the whole collection.  Writes invalidate these entries,
        # so cached values never hide newly indexed or deleted documents.
        self._signature_cache: Dict[str, str] = {}
        self._count_cache: Dict[str, int] = {}
        self._documents_cache: Dict[str, List[Dict]] = {}

    def _invalidate_patient_cache(self, patient_id: str) -> None:
        """Drop cached read results for one patient after a write."""
        self._signature_cache.pop(patient_id, None)
        self._count_cache.pop(patient_id, None)
        self._documents_cache.pop(patient_id, None)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _require_patient_id(patient_id: str):
        if not isinstance(patient_id, str) or not patient_id.strip():
            raise ValueError("A non-empty patient_id is required for every vector operation.")
        return patient_id.strip()

    def _where(self, patient_id: str, document_ids: Optional[Iterable[str]] = None) -> Dict:
        patient_id = self._require_patient_id(patient_id)
        document_ids = [str(doc_id) for doc_id in (document_ids or []) if doc_id]
        if not document_ids:
            return {"patient_id": patient_id}
        return {
            "$and": [
                {"patient_id": patient_id},
                {"document_id": {"$in": document_ids}},
            ]
        }

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def upsert_chunks(self, patient_id: str, chunks: List[Dict], embeddings) -> int:
        """
        Insert or update chunks.  Chunk IDs are stable, so re-running an
        unchanged index updates existing rows instead of creating duplicates.
        """
        patient_id = self._require_patient_id(patient_id)
        chunks = list(chunks or [])
        if not chunks:
            return 0

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must match.")

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for chunk in chunks:
            if str(chunk.get("patient_id", "")) != patient_id:
                raise ValueError(
                    "Refusing to index a chunk whose patient_id does not match "
                    "the active patient."
                )
            chunk_id = str(chunk.get("chunk_id", "") or "")
            if not chunk_id:
                raise ValueError("Every chunk must have a chunk_id.")
            ids.append(chunk_id)
            documents.append(str(chunk.get("text", "")))
            metadatas.append(_to_chroma_metadata(chunk))

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as error:
            logger.error("ChromaDB upsert failed: %s", error)
            raise VectorStoreError(f"Could not store chunks in ChromaDB: {error}") from error

        self._invalidate_patient_cache(patient_id)
        logger.info("Upserted %d chunk(s) for patient %s", len(ids), patient_id)
        return len(ids)

    def delete_document(self, patient_id: str, document_id: str) -> bool:
        """Delete all chunks belonging to one document owned by a patient."""
        patient_id = self._require_patient_id(patient_id)
        if not document_id:
            return False
        where = self._where(patient_id, [document_id])
        existing = self._collection.get(where=where, include=["metadatas"])
        if not existing.get("ids"):
            return False
        self._collection.delete(where=where)
        self._invalidate_patient_cache(patient_id)
        logger.info("Deleted document %s for patient %s", document_id, patient_id)
        return True

    def delete_patient(self, patient_id: str) -> int:
        """Delete every indexed chunk for one patient."""
        patient_id = self._require_patient_id(patient_id)
        existing = self._collection.get(where={"patient_id": patient_id}, include=["metadatas"])
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(where={"patient_id": patient_id})
        self._invalidate_patient_cache(patient_id)
        logger.info("Deleted %d chunk(s) for patient %s", len(ids), patient_id)
        return len(ids)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_document_metadata(self, patient_id: str, document_id: str) -> Optional[Dict]:
        """Return metadata for the first chunk of a document, if present."""
        patient_id = self._require_patient_id(patient_id)
        if not document_id:
            return None
        result = self._collection.get(
            where=self._where(patient_id, [document_id]),
            include=["metadatas"],
            limit=1,
        )
        metadatas = result.get("metadatas") or []
        return dict(metadatas[0]) if metadatas else None

    def document_exists(self, patient_id: str, document_id: str, content_hash: Optional[str] = None) -> bool:
        """Check whether a document is already indexed (optionally by hash)."""
        metadata = self.get_document_metadata(patient_id, document_id)
        if metadata is None:
            return False
        if content_hash and metadata.get("content_hash") != content_hash:
            return False
        return True

    def list_documents(self, patient_id: str) -> List[Dict]:
        """List distinct documents for one patient."""
        patient_id = self._require_patient_id(patient_id)
        cached = self._documents_cache.get(patient_id)
        if cached is not None:
            # Return copies so callers cannot mutate the cached entries.
            return [dict(item) for item in cached]
        result = self._collection.get(
            where={"patient_id": patient_id},
            include=["metadatas"],
        )
        metadatas = result.get("metadatas") or []
        documents: Dict[str, Dict] = {}

        for metadata in metadatas:
            document_id = metadata.get("document_id", "")
            if not document_id:
                continue
            if document_id not in documents:
                documents[document_id] = {
                    "document_id": document_id,
                    "filename": metadata.get("filename", "Unknown document"),
                    "file_type": metadata.get("file_type", ""),
                    "upload_timestamp": metadata.get("upload_timestamp", ""),
                    "document_date": metadata.get("document_date", ""),
                    "content_hash": metadata.get("content_hash", ""),
                    "chunk_count": 0,
                }
            documents[document_id]["chunk_count"] += 1

        ordered = sorted(documents.values(), key=lambda item: item["filename"].lower())
        self._documents_cache[patient_id] = ordered
        return [dict(item) for item in ordered]

    def patient_signature(self, patient_id: str) -> str:
        """A stable signature of a patient's index, used to cache BM25."""
        patient_id = self._require_patient_id(patient_id)
        cached = self._signature_cache.get(patient_id)
        if cached is not None:
            return cached
        result = self._collection.get(
            where={"patient_id": patient_id},
            include=["metadatas"],
        )
        parts = []
        for metadata in result.get("metadatas") or []:
            parts.append(
                f"{metadata.get('document_id','')}:{metadata.get('content_hash','')}:"
                f"{metadata.get('chunk_id','')}"
            )
        parts.sort()
        signature = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
        self._signature_cache[patient_id] = signature
        return signature

    def get_all_chunks(self, patient_id: str, document_ids: Optional[Iterable[str]] = None) -> List[Dict]:
        """Return all chunks for a patient (optionally limited to documents)."""
        patient_id = self._require_patient_id(patient_id)
        where = self._where(patient_id, document_ids)
        result = self._collection.get(
            where=where,
            include=["documents", "metadatas"],
        )

        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []

        chunks: List[Dict] = []
        for index, chunk_id in enumerate(ids):
            document = documents[index] if index < len(documents) else ""
            metadata = metadatas[index] if index < len(metadatas) else {}
            if metadata.get("patient_id") != patient_id:
                logger.warning("Discarding chunk with mismatched patient_id")
                continue
            chunks.append(_from_chroma_row(chunk_id, document, metadata))
        return chunks

    def query(
        self,
        patient_id: str,
        query_embedding,
        top_k: int = 5,
        document_ids: Optional[Iterable[str]] = None,
    ) -> List[Dict]:
        """Run a cosine similarity search scoped to one patient."""
        patient_id = self._require_patient_id(patient_id)
        top_k = max(1, int(top_k))

        embedding = np.asarray(query_embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding.reshape(1, -1)

        try:
            result = self._collection.query(
                query_embeddings=embedding.tolist(),
                n_results=top_k,
                where=self._where(patient_id, document_ids),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as error:
            logger.error("ChromaDB query failed: %s", error)
            raise VectorStoreError(f"Vector search failed: {error}") from error

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        rows: List[Dict] = []
        for index, chunk_id in enumerate(ids):
            metadata = metadatas[index] if index < len(metadatas) else {}
            if metadata.get("patient_id") != patient_id:
                logger.warning("Discarding a search result from another patient")
                continue
            document = documents[index] if index < len(documents) else ""
            distance = distances[index] if index < len(distances) else None
            rows.append(_from_chroma_row(chunk_id, document, metadata, distance))

        return rows

    def count(self, patient_id: Optional[str] = None) -> int:
        """Count chunks globally or for one patient."""
        if patient_id is None:
            return int(self._collection.count())
        patient_id = self._require_patient_id(patient_id)
        cached = self._count_cache.get(patient_id)
        if cached is not None:
            return cached
        result = self._collection.get(
            where={"patient_id": patient_id},
            include=["metadatas"],
        )
        count = len(result.get("ids") or [])
        self._count_cache[patient_id] = count
        return count


def get_vector_store(persist_dir: Optional[str] = None, collection_name: Optional[str] = None) -> VectorStore:
    """Convenience factory used by the Streamlit app."""
    return VectorStore(persist_dir=persist_dir, collection_name=collection_name)
