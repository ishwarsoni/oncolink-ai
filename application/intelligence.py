"""Unified document intelligence orchestrator.

One ingestion result feeds two independent capabilities:

    ingestion -> canonical ClinicalDocument(s)
                 |-> structured clinical intelligence (LLM extraction, ...)
                 `-> evidence retrieval and RAG (chunk, embed, answer)

The orchestrator tracks the status of each stage separately.  A failure in
one branch is recorded on that branch only and never corrupts the other
branch's data or status.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from application.clinical import (
    ClinicalRun,
    ClinicalSummary,
    extract_clinical,
    harmonize_clinical,
)
from application.ingestion import IngestionResult, ingest_uploads
from application.rag import RAGService
from application.status import StageStatus
from domain.documents import ClinicalDocument, generate_patient_id

logger = logging.getLogger(__name__)

INGESTION = "ingestion"
CLINICAL = "clinical"
RAG = "rag"


class DocumentIntelligenceService:
    """Coordinate ingestion, clinical extraction and RAG for one patient case."""

    def __init__(
        self,
        patient_id: str | None = None,
        rag_service: RAGService | None = None,
    ):
        self.patient_id = patient_id or generate_patient_id()
        self.rag = rag_service or RAGService()

        self.ingestion: IngestionResult | None = None
        self.clinical_run: ClinicalRun | None = None
        self.harmonization: ClinicalSummary | None = None

        self.statuses: dict[str, StageStatus] = {
            INGESTION: StageStatus(INGESTION),
            CLINICAL: StageStatus(CLINICAL),
            RAG: StageStatus(RAG),
        }

    # ------------------------------------------------------------------
    # Shared data
    # ------------------------------------------------------------------
    @property
    def documents(self) -> tuple[ClinicalDocument, ...]:
        """Canonical documents shared by both branches."""
        return self.ingestion.documents if self.ingestion else ()

    @property
    def display_records(self) -> tuple[dict[str, Any], ...]:
        """Legacy display records, including files that failed to load."""
        return self.ingestion.display_records if self.ingestion else ()

    def stage_status(self, stage: str) -> StageStatus:
        """Return the status object for one stage."""
        return self.statuses[stage]

    def snapshot(self) -> dict[str, Any]:
        """Return a serialisable view of the current workflow state."""
        return {
            "patient_id": self.patient_id,
            "document_ids": [document.document_id for document in self.documents],
            "document_count": len(self.documents),
            "statuses": {
                name: status.as_dict() for name, status in self.statuses.items()
            },
        }

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def ingest(self, uploaded_files: Iterable[Any]) -> IngestionResult:
        """Read uploads once and make the canonical documents available.

        New documents invalidate downstream clinical/RAG results because they
        describe a different document set, but this is a lifecycle decision,
        not a failure of either branch.
        """
        status = self.statuses[INGESTION].mark_running("Reading uploaded documents")
        try:
            result = ingest_uploads(uploaded_files, self.patient_id)
        except Exception as error:
            logger.exception("Document ingestion failed")
            status.mark_failed(str(error))
            raise

        self.ingestion = result
        self.clinical_run = None
        self.harmonization = None
        self.statuses[CLINICAL] = StageStatus(CLINICAL)
        self.statuses[RAG] = StageStatus(RAG)
        self.rag.set_documents(result.documents)

        if result.has_documents:
            message = f"Read {len(result.documents)} document(s)."
            if result.errors:
                message += f" {len(result.errors)} file(s) could not be read."
            status.mark_succeeded(message)
        elif result.errors:
            status.mark_failed("; ".join(result.errors))
        else:
            status.mark_skipped("No readable documents were uploaded.")

        return result

    # ------------------------------------------------------------------
    # Clinical branch
    # ------------------------------------------------------------------
    def run_clinical(self, documents: Sequence[ClinicalDocument] | None = None) -> ClinicalRun:
        """Run structured extraction without affecting the RAG branch."""
        status = self.statuses[CLINICAL].mark_running("Extracting structured data")
        documents = list(documents if documents is not None else self.documents)

        if not documents:
            status.mark_skipped("No documents are available for extraction.")
            empty = ClinicalRun(
                combined={
                    "success": False,
                    "data": None,
                    "errors": ["No readable documents are available for extraction."],
                    "warnings": [],
                }
            )
            self.clinical_run = empty
            return empty

        try:
            run = extract_clinical(documents)
        except Exception as error:  # one branch must not break the other
            logger.exception("Structured clinical extraction failed")
            status.mark_failed(str(error))
            failed = ClinicalRun(
                combined={
                    "success": False,
                    "data": None,
                    "errors": [str(error)],
                    "warnings": [],
                }
            )
            self.clinical_run = failed
            return failed

        self.clinical_run = run
        self.harmonization = None

        if run.success:
            status.mark_succeeded("Structured clinical data extracted.")
        else:
            errors = run.errors or ("Extraction produced no clinical data.",)
            status.mark_failed("; ".join(errors))

        return run

    def harmonize(self) -> ClinicalSummary:
        """Harmonize per-document extractions and generate a patient summary.

        Harmonization is part of the clinical branch.  It never changes the
        RAG branch status.
        """
        status = self.statuses[CLINICAL]

        if self.clinical_run is None:
            status.mark_skipped("Run structured extraction before harmonization.")
            return ClinicalSummary(harmonized={}, conflicts={}, summary={})

        try:
            result = harmonize_clinical(self.clinical_run.per_document)
        except Exception as error:
            logger.exception("Harmonization failed")
            status.mark_failed(str(error))
            return ClinicalSummary(harmonized={}, conflicts={}, summary={})

        self.harmonization = result
        if result.harmonized.get("success"):
            status.mark_succeeded(
                f"Harmonized {result.harmonized.get('documents_merged', 0)} document(s)."
            )
        return result

    # ------------------------------------------------------------------
    # RAG branch
    # ------------------------------------------------------------------
    def ensure_rag_index(self, force: bool = False) -> dict[str, Any]:
        """Index the shared documents for retrieval, independently of clinical."""
        status = self.statuses[RAG].mark_running("Indexing documents for retrieval")

        self.rag.set_documents(self.documents)
        try:
            result = self.rag.ensure_index(force=force)
        except Exception as error:
            logger.exception("RAG indexing failed")
            status.mark_failed(str(error))
            return {
                "success": False,
                "indexed": 0,
                "updated": 0,
                "skipped": 0,
                "chunks_indexed": 0,
                "errors": [str(error)],
                "warnings": [],
                "documents": [],
            }

        if self.ingestion and self.ingestion.warnings:
            merged = list(result.get("warnings", []))
            merged.extend(self.ingestion.warnings)
            result["warnings"] = merged

        if result.get("success"):
            status.mark_succeeded(
                f"Index ready: {result.get('indexed', 0)} new, "
                f"{result.get('updated', 0)} updated, "
                f"{result.get('skipped', 0)} unchanged."
            )
        else:
            status.mark_failed(
                "; ".join(result.get("errors") or ["RAG indexing failed."])
            )
        return result

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        mode: str = "hybrid",
        rerank: bool = False,
        score_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Answer one question from the current patient's indexed evidence."""
        status = self.statuses[RAG].mark_running("Answering from retrieved evidence")
        try:
            result = self.rag.ask(
                patient_id=self.patient_id,
                question=question,
                top_k=top_k,
                mode=mode,
                rerank=rerank,
                score_threshold=score_threshold,
            )
        except Exception as error:
            logger.exception("RAG question failed")
            status.mark_failed(str(error))
            raise

        if result.get("success"):
            status.mark_succeeded("Answer generated with validated citations.")
        else:
            status.mark_failed(result.get("error") or "No grounded answer was produced.")
        return result

    def index_status(self):
        """Return the current patient's index summary."""
        return self.rag.index_status(self.patient_id)

    def delete_rag_index(self) -> int:
        """Delete this patient's indexed chunks and reset the RAG cache."""
        deleted = self.rag.remove_index(self.patient_id)
        self.statuses[RAG] = StageStatus(RAG)
        self.statuses[RAG].mark_skipped("Index deleted for this case.")
        return deleted

    # ------------------------------------------------------------------
    # Case lifecycle
    # ------------------------------------------------------------------
    def reset(self, patient_id: str | None = None) -> "DocumentIntelligenceService":
        """Start a clean case, keeping the same lazy RAG stack if possible."""
        self.patient_id = patient_id or generate_patient_id()
        self.ingestion = None
        self.clinical_run = None
        self.harmonization = None
        self.statuses = {
            INGESTION: StageStatus(INGESTION),
            CLINICAL: StageStatus(CLINICAL),
            RAG: StageStatus(RAG),
        }
        self.rag.set_documents(())
        return self
