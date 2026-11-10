"""Per-stage processing status for the document intelligence workflow.

Ingestion, structured clinical extraction and RAG indexing fail for different
reasons and must be reported independently.  A failed clinical extraction
therefore never marks the RAG index as failed, and vice versa.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StageState(str, Enum):
    """Lifecycle states shared by every processing stage."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageStatus:
    """Status of one independent stage of the pipeline."""

    stage: str
    state: StageState = StageState.PENDING
    message: str = ""
    errors: list[str] = field(default_factory=list)

    @property
    def successful(self) -> bool:
        """True only when the stage finished successfully."""
        return self.state is StageState.SUCCEEDED

    def mark_running(self, message: str = "") -> "StageStatus":
        """Record that the stage has started."""
        self.state = StageState.RUNNING
        self.message = message
        self.errors = []
        return self

    def mark_succeeded(self, message: str = "") -> "StageStatus":
        """Record a successful completion and clear previous errors."""
        self.state = StageState.SUCCEEDED
        self.message = message
        self.errors = []
        return self

    def mark_failed(self, error: str) -> "StageStatus":
        """Record a failure without affecting any other stage."""
        self.state = StageState.FAILED
        self.message = error
        self.errors = [error]
        return self

    def mark_skipped(self, reason: str) -> "StageStatus":
        """Record that the stage did not run and why."""
        self.state = StageState.SKIPPED
        self.message = reason
        self.errors = []
        return self

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for exports and tests."""
        return {
            "stage": self.stage,
            "state": self.state.value,
            "message": self.message,
            "errors": list(self.errors),
        }
