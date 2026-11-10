"""Canonical document models shared by every OncoLink capability.

The structured-extraction pipeline and the RAG pipeline must see exactly the
same document: the same patient ID, the same stable document ID, the same
text, the same 1-based page boundaries, and the same content hash.  Building
two slightly different representations was the main source of duplicated
reads, duplicated hashing and index/citation drift in the original code.

This module is intentionally free of Streamlit, ChromaDB and LLM imports so
it can be unit-tested in isolation and reused by both branches.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

# ---------------------------------------------------------------------------
# Stable identifiers and hashes
# ---------------------------------------------------------------------------


def generate_patient_id() -> str:
    """Create a random, non-identifying case ID for one browser session.

    Patient names are never used as identifiers because they would leak into
    logs, ChromaDB metadata and citations.
    """
    return f"case_{uuid.uuid4().hex[:12]}"


def safe_filename(filename: Any) -> str:
    """Reduce a user-supplied filename to a safe display/citation value.

    Directory components and control characters are removed so a crafted
    upload name cannot be used for path traversal or log injection.
    """
    if not filename:
        return "unknown"
    # Normalise both separator styles so a crafted Windows-style name is also
    # reduced to its basename when the app runs on Linux.
    name = str(filename).replace("\\", "/")
    name = Path(name).name
    name = re.sub(r"[\x00-\x1f\x7f]", "", name).strip()
    return name or "unknown"


def document_content_hash(text: str) -> str:
    """Return the SHA-256 hash used to detect unchanged document content."""
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


def build_document_id(patient_id: str, filename: str) -> str:
    """Derive a stable document ID from the patient and the display filename.

    Re-uploading a corrected version of the same file therefore updates the
    existing document instead of creating a duplicate.
    """
    patient_id = str(patient_id or "").strip()
    if not patient_id:
        raise ValueError("A patient_id is required to build a document ID.")
    return hashlib.sha256(f"{patient_id}|{safe_filename(filename)}".encode("utf-8")).hexdigest()[:16]


def index_signature(documents: Iterable[Any]) -> str:
    """Return a short signature for a set of documents.

    The signature covers each document ID and content hash, so it changes
    whenever a document is added, removed or edited.  Callers use it to decide
    whether a cached indexing result is still valid.
    """
    parts = sorted(
        f"{_field(doc, 'document_id', '')}:{_field(doc, 'content_hash', '')}"
        for doc in documents or []
    )
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Document dates
# ---------------------------------------------------------------------------

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),  # YYYY-MM-DD
    re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"),  # DD-MM-YYYY
    re.compile(r"\b(\d{1,2})[-\s]([A-Za-z]{3,9})[-\s](\d{4})\b"),  # DD-Mon-YYYY
    re.compile(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"),  # Month DD, YYYY
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


def _iso_date(year: Any, month: Any, day: Any) -> Optional[str]:
    """Build an ISO date, returning None for impossible calendar values."""
    try:
        return datetime(int(year), int(month), int(day)).date().isoformat()
    except (TypeError, ValueError):
        return None


def extract_document_date(text: str) -> str:
    """Extract an explicit document date and normalise it to ISO when possible.

    Only labelled dates (date, visit date, exam date, ...) are considered, so
    dates of birth or treatment dates inside prose are not mistaken for the
    document date.  An empty string means "not found"; the code never guesses.
    """
    if not text:
        return ""

    for label_match in _DATE_LABEL_PATTERN.finditer(text):
        window = text[label_match.end(): label_match.end() + 40]
        for pattern in _DATE_PATTERNS:
            match = pattern.search(window)
            if not match:
                continue
            first, second, third = match.groups()

            if pattern is _DATE_PATTERNS[0]:
                iso = _iso_date(first, second, third)
            elif pattern is _DATE_PATTERNS[1]:
                iso = _iso_date(third, second, first)
            elif pattern is _DATE_PATTERNS[2]:
                month = _MONTHS.get(second.lower())
                iso = _iso_date(third, month, first) if month else None
            else:
                month = _MONTHS.get(first.lower())
                iso = _iso_date(third, month, second) if month else None

            if iso:
                return iso
            return match.group(0).strip()
    return ""


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DocumentPage:
    """One page (or the single logical page of a DOCX/TXT file)."""

    page_number: Optional[int]
    text: str

    @property
    def is_paginated(self) -> bool:
        """True when the source format carries real page numbers."""
        return self.page_number is not None

    def as_dict(self) -> dict[str, Any]:
        return {"page_number": self.page_number, "text": self.text}


@dataclass(frozen=True)
class ClinicalDocument:
    """A document that has been read once and can feed both pipelines.

    Instances are immutable so one branch cannot accidentally rewrite the
    text, pages or identifiers another branch is using.
    """

    patient_id: str
    document_id: str
    filename: str
    file_type: str
    text: str
    pages: tuple[DocumentPage, ...] = field(default_factory=tuple)
    content_hash: str = ""
    upload_timestamp: str = ""
    document_date: str = ""

    # -- construction ---------------------------------------------------
    @classmethod
    def from_processed_file(
        cls,
        processed_file: Mapping[str, Any],
        patient_id: str,
        upload_timestamp: Optional[str] = None,
    ) -> "ClinicalDocument":
        """Build a canonical document from a loader result.

        ``processed_file`` is the dictionary returned by
        ``backend.document_loader.process_uploaded_file``.  Successful loads
        only; callers keep failures in the display record instead.
        """
        patient_id = str(patient_id or "").strip()
        if not patient_id:
            raise ValueError("A patient_id is required when building a document.")

        filename = safe_filename(processed_file.get("filename", "unknown"))
        text = processed_file.get("text", "") or ""
        raw_pages = processed_file.get("pages") or []
        pages = tuple(
            DocumentPage(page.get("page_number"), page.get("text", "") or "")
            for page in raw_pages
            if isinstance(page, Mapping)
        )
        if not pages and text:
            pages = (DocumentPage(None, text),)

        return cls(
            patient_id=patient_id,
            document_id=build_document_id(patient_id, filename),
            filename=filename,
            file_type=str(processed_file.get("file_type", "")),
            text=text,
            pages=pages,
            content_hash=document_content_hash(text),
            upload_timestamp=upload_timestamp
            or datetime.now(timezone.utc).isoformat(timespec="seconds"),
            document_date=extract_document_date(text),
        )

    # -- projections ----------------------------------------------------
    @property
    def is_empty(self) -> bool:
        """True when the document has no usable text."""
        return not self.text.strip()

    def as_index_record(self) -> dict[str, Any]:
        """Return the mapping consumed by the chunker and vector store.

        The chunker and ChromaDB layer operate on plain dictionaries because
        they are also used directly by the offline tests.  Keeping the
        projection in one place guarantees both pipelines see identical data.
        """
        return {
            "patient_id": self.patient_id,
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "text": self.text,
            "pages": [page.as_dict() for page in self.pages],
            "content_hash": self.content_hash,
            "upload_timestamp": self.upload_timestamp,
            "document_date": self.document_date,
        }

    def as_processed_file(self) -> dict[str, Any]:
        """Return the legacy loader shape used by UI display and extractor."""
        return {
            "filename": self.filename,
            "text": self.text,
            "file_type": self.file_type,
            "success": True,
            "pages": [page.as_dict() for page in self.pages],
        }


def _field(document: Any, key: str, default: Any = None) -> Any:
    """Read a field from a ClinicalDocument or a legacy mapping."""
    if isinstance(document, ClinicalDocument):
        return getattr(document, key, default)
    if isinstance(document, Mapping):
        return document.get(key, default)
    return getattr(document, key, default)
