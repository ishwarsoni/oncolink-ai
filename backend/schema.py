"""
schema.py - Single source of truth for the clinical data structure.

This module defines the Pydantic models used to validate LLM output and the
display metadata used by the UI and exports.  Keeping both in one place
prevents the runtime model and the documentation from drifting apart.

Field metadata is derived from ``ClinicalData.model_fields`` rather than
re-declared, so adding a field only requires one edit.
"""

from typing import Any, Dict, List, Optional, get_args, get_origin

from pydantic import BaseModel, Field


class BiomarkerItem(BaseModel):
    """One biomarker measurement (e.g. ER Positive, Ki-67 25%)."""

    name: str = ""
    value: str = ""
    status: Optional[str] = None


class ClinicalData(BaseModel):
    """
    Structured clinical data extracted from oncology documents.

    Optional fields use ``None`` to mean "not found in the documents"; they
    never use an empty string, which would be indistinguishable from a value
    the model actually read.
    """

    # ---- Patient identity ----
    patient_name: Optional[str] = Field(
        default=None, description="Full name of the patient"
    )
    age: Optional[int] = Field(
        default=None, ge=0, le=150, description="Patient's age in years"
    )
    gender: Optional[str] = Field(
        default=None, description="Patient's gender (Male/Female/Other)"
    )

    # ---- Cancer information ----
    diagnosis: Optional[str] = Field(
        default=None, description="Complete diagnosis description"
    )
    cancer_type: Optional[str] = Field(
        default=None, description="Specific type of cancer"
    )
    cancer_stage: Optional[str] = Field(
        default=None, description="Stage including TNM classification if available"
    )

    # ---- Biomarkers ----
    biomarkers: List[BiomarkerItem] = Field(
        default_factory=list,
        description="List of biomarkers with name, value, and status",
    )

    # ---- Treatment ----
    current_medication: Optional[str] = Field(
        default=None, description="Current medications the patient is taking"
    )
    previous_treatment: Optional[str] = Field(
        default=None, description="Treatments the patient has already received"
    )

    # ---- Performance status ----
    ecog_score: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="ECOG performance status (0-5) or null if not found",
    )

    # ---- Plan ----
    follow_up_plan: Optional[str] = Field(
        default=None, description="Recommended follow-up schedule"
    )
    next_steps: Optional[str] = Field(
        default=None, description="Recommended next actions or treatment plan"
    )

    # ---- Adverse events ----
    adverse_events: Optional[str] = Field(
        default=None,
        description="Side effects, toxicities, or adverse reactions to treatment",
    )


# Display labels are presentation data, so they cannot be derived from the
# model.  They are intentionally kept next to the model they describe.
FIELD_LABELS = {
    "patient_name": "Patient Name",
    "age": "Age",
    "gender": "Gender",
    "diagnosis": "Diagnosis",
    "cancer_type": "Cancer Type",
    "cancer_stage": "Cancer Stage",
    "biomarkers": "Biomarkers",
    "current_medication": "Current Medication",
    "previous_treatment": "Previous Treatment",
    "ecog_score": "ECOG Score",
    "follow_up_plan": "Follow-up Plan",
    "next_steps": "Next Steps",
    "adverse_events": "Adverse Events",
}

# Fields that should normally be present in a usable patient record.  Missing
# values are reported as warnings, not hard validation failures.
REQUIRED_FIELDS = (
    "patient_name",
    "age",
    "gender",
    "diagnosis",
    "cancer_type",
    "cancer_stage",
)


def _python_type_name(annotation: Any) -> str:
    """Map a Pydantic field annotation to the legacy schema type name."""
    origin = get_origin(annotation)
    if origin in (list, List):
        return "list"

    # Optional[X] is Union[X, None]; the payload type is the first argument.
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    base_type = args[0] if args else annotation
    if base_type in (int, float):
        return "number"
    return "string"


def _build_clinical_schema() -> Dict[str, Dict[str, Any]]:
    """Derive the documented field schema from the Pydantic model."""
    schema: Dict[str, Dict[str, Any]] = {}
    for field_name, model_field in ClinicalData.model_fields.items():
        schema[field_name] = {
            "type": _python_type_name(model_field.annotation),
            "label": FIELD_LABELS.get(field_name, field_name.replace("_", " ").title()),
            "description": model_field.description or "",
            "required": field_name in REQUIRED_FIELDS,
        }
    return schema


# Reference metadata for documentation, tests, and future export formats.
# It is generated from ClinicalData, so it can never contradict validation.
CLINICAL_SCHEMA = _build_clinical_schema()


def get_field_label(field_name: str) -> str:
    """Return the user-friendly label for a clinical field."""
    if field_name in CLINICAL_SCHEMA:
        return CLINICAL_SCHEMA[field_name]["label"]
    return field_name.replace("_", " ").title()


def get_required_fields() -> List[str]:
    """Return the field names that are expected in a complete record."""
    return [name for name in REQUIRED_FIELDS if name in ClinicalData.model_fields]


def create_validation_result(
    valid: bool,
    data: Optional[Dict],
    errors: Optional[List[str]] = None,
    warnings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a validation result with a consistent shape."""
    return {
        "valid": valid,
        "data": data,
        "errors": errors or [],
        "warnings": warnings or [],
    }
