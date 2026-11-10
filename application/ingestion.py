"""Ingestion workflow: read every upload once and build canonical documents.

Both capabilities (structured extraction and RAG) consume the documents
returned here, so a file is never read, parsed or hashed twice for one
processing operation.  Files that cannot be read are kept as display records
with an error message; they do not become canonical documents and cannot
reach either clinical pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from backend.document_loader import process_uploaded_file
from domain.documents import ClinicalDocument, safe_filename

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of reading one batch of uploaded files.

    ``documents`` is the canonical shared representation.  ``display_records``
    keeps the legacy loader shape for the extracted-text view, including
    failures, so the UI can explain exactly which upload was rejected.
    """

    patient_id: str
    documents: tuple[ClinicalDocument, ...] = ()
    display_records: tuple[dict[str, Any], ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def has_documents(self) -> bool:
        """True when at least one upload produced usable text."""
        return bool(self.documents)

    @property
    def failed_filenames(self) -> tuple[str, ...]:
        """Filenames that could not be read."""
        return tuple(
            record["filename"]
            for record in self.display_records
            if not record.get("success")
        )


def ingest_uploads(
    uploaded_files: Iterable[Any],
    patient_id: str,
    reader: Callable[[Any], dict[str, Any]] = process_uploaded_file,
) -> IngestionResult:
    """Read each uploaded file exactly once and build canonical documents.

    Args:
        uploaded_files: Streamlit ``UploadedFile`` objects, or any object with
            ``.name`` and ``.read()``.
        patient_id: Non-identifying case ID that scopes both pipelines.
        reader: Loader function, injectable for tests.

    Returns:
        An :class:`IngestionResult` with canonical documents, display records,
        and any per-file errors.
    """
    patient_id = str(patient_id or "").strip()
    if not patient_id:
        raise ValueError("A patient_id is required to ingest documents.")

    upload_timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    display_records: list[dict[str, Any]] = []
    documents: list[ClinicalDocument] = []
    errors: list[str] = []
    warnings: list[str] = []
    seen_document_ids: set[str] = set()

    for uploaded_file in uploaded_files or []:
        record = _read_one(uploaded_file, reader, errors)
        display_records.append(record)

        if not record["success"]:
            continue
        if not record["text"].strip():
            warnings.append(
                f"{record['filename']}: no extractable text "
                "(the file may be a scanned document)."
            )
            continue

        document = _build_document(record, patient_id, upload_timestamp, errors)
        if document is None:
            continue
        if document.document_id in seen_document_ids:
            warnings.append(
                f"{document.filename}: duplicate upload ignored for indexing."
            )
            continue

        seen_document_ids.add(document.document_id)
        documents.append(document)

    return IngestionResult(
        patient_id=patient_id,
        documents=tuple(documents),
        display_records=tuple(display_records),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _read_one(
    uploaded_file: Any,
    reader: Callable[[Any], dict[str, Any]],
    errors: list[str],
) -> dict[str, Any]:
    """Read one upload and return the legacy display record.

    A reader exception is converted into a failed record: one malformed file
    must never abort the whole batch.
    """
    raw_name = getattr(uploaded_file, "name", "unknown")
    filename = safe_filename(raw_name)

    try:
        result = reader(uploaded_file)
    except Exception as error:  # defensive: readers should report, not raise
        logger.exception("Could not read uploaded file %s", filename)
        message = f"Could not read {filename}: {error}"
        errors.append(message)
        return _failed_record(filename, message, "UNKNOWN", [])

    text = result.get("text", "") or ""
    file_type = result.get("file_type", "UNKNOWN")
    pages = result.get("pages") or []
    success = bool(result.get("success")) and not text.startswith("Error")

    if not success and text:
        errors.append(f"{filename}: {text}")

    return {
        "filename": filename,
        "text": text,
        "file_type": file_type,
        "success": success,
        "pages": pages,
    }


def _failed_record(
    filename: str, message: str, file_type: str, pages: list[dict]
) -> dict[str, Any]:
    """Build a display record for a file that could not be read."""
    return {
        "filename": filename,
        "text": message,
        "file_type": file_type,
        "success": False,
        "pages": pages,
    }


def _build_document(
    record: dict[str, Any],
    patient_id: str,
    upload_timestamp: str,
    errors: list[str],
) -> ClinicalDocument | None:
    """Convert a successful loader record into a canonical document."""
    try:
        return ClinicalDocument.from_processed_file(
            record,
            patient_id,
            upload_timestamp=upload_timestamp,
        )
    except Exception as error:  # defensive: keep other files usable
        logger.exception("Could not build canonical document for %s", record["filename"])
        errors.append(f"{record['filename']}: {error}")
        return None
