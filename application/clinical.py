"""Structured clinical intelligence workflow.

This module owns the order of the clinical branch:

    per-document + combined extraction -> harmonization -> conflicts -> summary

The LLM prompt, schema, validator and merge rules stay in ``backend``; this
service only coordinates them and translates failures into an explicit result.
A failed extraction is never turned into empty clinical data.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from backend.conflict_detector import detect_conflicts
from backend.extractor import extract_each_document, extract_from_multiple_documents
from backend.harmonizer import harmonize_extractions
from backend.summarizer import generate_patient_summary
from infrastructure.llm import create_nvidia_client

logger = logging.getLogger(__name__)


def _dedupe(values: Iterable[str]) -> tuple[str, ...]:
    """Remove duplicate messages while preserving their original order."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


@dataclass(frozen=True)
class ClinicalRun:
    """Result of running structured extraction over a document set."""

    combined: dict[str, Any]
    per_document: tuple[dict[str, Any], ...] = ()

    @property
    def success(self) -> bool:
        """True when the combined extraction produced validated data."""
        return bool(self.combined.get("success")) and self.combined.get("data") is not None

    @property
    def errors(self) -> tuple[str, ...]:
        """Combined errors followed by per-document errors, without repeats."""
        errors = list(self.combined.get("errors") or [])
        for result in self.per_document:
            filename = result.get("filename", "unknown")
            for error in result.get("errors") or []:
                errors.append(f"{filename}: {error}")
        return _dedupe(errors)

    @property
    def warnings(self) -> tuple[str, ...]:
        """Combined warnings followed by per-document warnings, without repeats."""
        warnings = list(self.combined.get("warnings") or [])
        for result in self.per_document:
            filename = result.get("filename", "unknown")
            for warning in result.get("warnings") or []:
                warnings.append(f"{filename}: {warning}")
        return _dedupe(warnings)


@dataclass(frozen=True)
class ClinicalSummary:
    """Harmonized record, conflicts and narrative summary."""

    harmonized: dict[str, Any]
    conflicts: dict[str, Any]
    summary: dict[str, Any]


def extract_clinical(
    documents: Sequence[Any], client: Any = None
) -> ClinicalRun:
    """Run combined and per-document extraction over canonical documents.

    Both passes share one NVIDIA client so a batch reuses a single connection
    pool.  The combined result powers the main structured view; per-document
    results preserve provenance for harmonization and conflict detection.
    """
    documents = list(documents or [])
    if client is None:
        # One client for both passes reuses connection pooling.  When no key
        # is configured we pass None through so each extractor reports the
        # configuration error in its normal result shape.
        try:
            client = create_nvidia_client()
        except Exception:
            client = None

    combined = extract_from_multiple_documents(documents, client=client)
    per_document = extract_each_document(documents, client=client)

    logger.info(
        "Clinical extraction finished: combined=%s, documents=%d",
        bool(combined.get("success")),
        len(per_document),
    )
    return ClinicalRun(combined=combined, per_document=tuple(per_document))


def harmonize_clinical(per_document: Iterable[dict[str, Any]]) -> ClinicalSummary:
    """Merge per-document extractions deterministically and build a summary."""
    per_document = list(per_document or [])

    harmonized = harmonize_extractions(per_document)
    conflicts = detect_conflicts(per_document)

    if harmonized.get("success"):
        summary = generate_patient_summary(
            harmonized.get("data"),
            harmonized.get("field_sources"),
            conflicts.get("conflicts", []),
        )
    else:
        summary = {}

    return ClinicalSummary(
        harmonized=harmonized,
        conflicts=conflicts,
        summary=summary,
    )
