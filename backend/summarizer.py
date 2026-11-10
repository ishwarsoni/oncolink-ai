"""
summarizer.py - Render a human-readable patient summary.

The summary is plain text so it can be displayed in the UI, downloaded as
TXT, or rendered to PDF without a template engine.  Formatting is kept in
small helpers so the layout stays consistent across sections.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LINE_WIDTH = 60
_TITLE_RULE = "=" * _LINE_WIDTH
_SECTION_RULE = "-" * _LINE_WIDTH
_ALERT_RULE = "!" * _LINE_WIDTH

# Text fields rendered as a bullet list of semicolon-separated items.
_LIST_SECTIONS = (
    ("current_medication", "CURRENT MEDICATIONS"),
    ("previous_treatment", "PREVIOUS TREATMENT"),
    ("follow_up_plan", "FOLLOW-UP PLAN"),
    ("next_steps", "NEXT STEPS"),
    ("adverse_events", "ADVERSE EVENTS"),
)


def generate_patient_summary(
    harmonized_data: Optional[Dict[str, Any]],
    field_sources: Optional[Dict[str, str]] = None,
    conflicts: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a text summary for a harmonized patient record."""
    if not harmonized_data or not isinstance(harmonized_data, dict):
        return {"success": False, "summary": None, "error": "No harmonized data available."}

    try:
        lines = _format_header(harmonized_data)
        lines.extend(_format_conflicts(conflicts))
        lines.extend(_format_diagnosis_section(harmonized_data))
        lines.extend(_format_biomarkers(harmonized_data))

        for field_name, title in _LIST_SECTIONS:
            lines.extend(_format_bullet_section(title, harmonized_data.get(field_name)))

        lines.append(_TITLE_RULE)
        logger.info("Patient summary generated successfully")
        return {"success": True, "summary": "\n".join(lines), "error": None}

    except Exception as error:  # defensive: a summary must never break the UI
        logger.error("Summary generation failed: %s", error)
        return {
            "success": False,
            "summary": None,
            "error": f"Summary generation failed: {error}",
        }


def _format_header(data: Dict[str, Any]) -> List[str]:
    """Render the title and patient identity line."""
    name = data.get("patient_name", "Unknown")
    age = data.get("age", "Unknown")
    gender = data.get("gender", "Unknown")
    return [
        _TITLE_RULE,
        "PATIENT SUMMARY",
        _TITLE_RULE,
        "",
        f"Patient: {name} | Age: {age} | Gender: {gender}",
        "",
    ]


def _format_conflicts(conflicts: Optional[List[Dict[str, Any]]]) -> List[str]:
    """Render high-severity conflicts at the top so they are not missed."""
    high_severity = [
        conflict for conflict in (conflicts or []) if conflict.get("severity") == "high"
    ]
    if not high_severity:
        return []

    lines = [_ALERT_RULE, "CONFLICTS DETECTED", _ALERT_RULE]
    for conflict in high_severity:
        field_label = conflict.get("field", "Unknown field").replace("_", " ").title()
        lines.append(f"  ! {field_label}: {conflict['description']}")
        for document, value in conflict.get("sources", []):
            lines.append(f"    {document}: {value}")
    lines.append("")
    return lines


def _format_diagnosis_section(data: Dict[str, Any]) -> List[str]:
    """Render diagnosis, cancer type, stage, and ECOG status."""
    lines = [_SECTION_RULE, "DIAGNOSIS & STAGING", _SECTION_RULE]

    for field_name, label in (
        ("diagnosis", "Diagnosis"),
        ("cancer_type", "Cancer Type"),
        ("cancer_stage", "Stage"),
    ):
        value = data.get(field_name)
        if value:
            lines.append(f"  {label}: {value}")

    ecog_score = data.get("ecog_score")
    if ecog_score is not None:
        lines.append(f"  ECOG Performance Status: {ecog_score}")

    lines.append("")
    return lines


def _format_biomarkers(data: Dict[str, Any]) -> List[str]:
    """Render one line per biomarker, including status when present."""
    biomarkers = data.get("biomarkers", [])
    if not biomarkers:
        return []

    lines = [_SECTION_RULE, "BIOMARKERS", _SECTION_RULE]
    for biomarker in biomarkers:
        name = biomarker.get("name", "Unknown")
        value = biomarker.get("value", "N/A")
        status = biomarker.get("status")
        suffix = f" ({status})" if status else ""
        lines.append(f"  {name}: {value}{suffix}")

    lines.append("")
    return lines


def _format_bullet_section(title: str, value: Optional[str]) -> List[str]:
    """Render a titled list of semicolon-separated items."""
    if not value:
        return []

    lines = [_SECTION_RULE, title, _SECTION_RULE]
    for item in value.split(";"):
        item = item.strip()
        if item:
            lines.append(f"  - {item}")

    lines.append("")
    return lines
