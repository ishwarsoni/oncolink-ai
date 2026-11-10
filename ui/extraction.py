"""
extraction.py - AI clinical extraction UI and structured-data display.

The extraction workflow (prompt, LLM call, validation, normalization) lives in
``application.clinical`` / ``backend.extractor``.  This module starts it,
stores the results in session state, and renders them.
"""

import json

import streamlit as st

from domain.clinical import get_field_label
from ui.components import (
    render_biomarkers,
    render_metric_grid,
    render_pills,
    render_section,
    render_text_blocks,
)
from ui.session import get_intelligence_service, resolve_api_key


def run_extraction():
    """Run the extraction workflow once and persist its results.

    Thin UI wrapper around ``DocumentIntelligenceService.run_clinical``; the
    prompt, validation and normalization logic stays in the application layer.
    """
    service = get_intelligence_service()

    with st.spinner("Calling NVIDIA AI... This may take 10-30 seconds."):
        run = service.run_clinical()

        st.session_state["extraction_result"] = run.combined
        st.session_state["per_doc_results"] = list(run.per_document)
        st.session_state["extraction_done"] = True

        # A new extraction invalidates the previous merged record.
        st.session_state["harmonized_data"] = None
        st.session_state["conflicts"] = None
        st.session_state["summary"] = None
        st.session_state["harmonization_done"] = False

    render_extraction_feedback(run.combined)
    return run.combined


def render_extraction_controls(
    show_heading=True,
    label="Run AI extraction",
    key="run_extraction_button",
):
    """Render the API warning and the extraction button.

    Returns the extraction result when the button was pressed in this run,
    otherwise ``None``.  Callers can re-read session state afterwards.  The
    label and widget key are parameters so a view can render a second control
    (for example inside a workflow expander) without duplicating a widget ID.
    """
    if "processed_files" not in st.session_state:
        return None

    if show_heading:
        render_section(
            "AI Clinical Extraction",
            "Extract a structured patient record from every uploaded document "
            "using the configured NVIDIA Nemotron model.",
        )

    api_ready = _render_api_warning()
    extraction_clicked = st.button(
        label,
        type="primary",
        key=key,
        disabled=not api_ready,
    )
    if extraction_clicked:
        return run_extraction()
    return None


def create_extraction_section():
    """Render the extraction button, feedback, and extracted data."""
    if "processed_files" not in st.session_state:
        return

    render_extraction_controls(show_heading=True)
    _render_extraction_results()


def _render_api_warning() -> bool:
    """Warn when no API key is configured and return whether one exists."""
    api_ready = resolve_api_key() is not None
    if not api_ready:
        st.warning(
            "NVIDIA API key not configured. Enter a key in the sidebar to enable "
            "AI extraction and answer generation. You can still index documents "
            "and inspect their extracted text."
        )
    return api_ready


def render_extraction_feedback(result):
    """Show success, validation warnings, or the errors from a run."""
    if result.get("success"):
        st.success("AI extraction completed successfully.")
        for warning in result.get("warnings", []):
            st.warning(warning)
        return

    errors = result.get("errors", [])
    if errors:
        st.error("Extraction failed:\n" + "\n".join(errors))
    else:
        st.error("Extraction failed with an unknown error.")


def _render_extraction_results():
    """Render the most recent extraction result, if there is one."""
    result = st.session_state.get("extraction_result")
    if not result:
        return

    if result.get("success"):
        validation = result.get("validation", {})
        if validation and validation.get("valid"):
            st.success("Data validated against the clinical schema.")

        normalization = result.get("normalization", {})
        if normalization and normalization.get("success"):
            st.caption("Values normalized for consistency.")

        display_structured_clinical_data(result.get("data"))
        return

    for error in result.get("errors", []):
        st.error(error)
    if result.get("raw_response"):
        with st.expander("Debug: raw model response"):
            st.text(result["raw_response"])


def _age_text(value):
    """Format age without inventing a value when it is absent."""
    if value is None:
        return None
    return f"{value} years"


def _render_field_conflicts(fields, conflict_fields):
    """Flag fields that the conflict detector reported for this record."""
    if not conflict_fields:
        return
    matched = [get_field_label(field) for field in fields if field in conflict_fields]
    if matched:
        render_pills([(f"Conflict reported: {', '.join(matched)}", "warn")])


def display_structured_clinical_data(data, show_export=True, conflict_fields=None):
    """Display clinical data as the sections used by the patient record.

    The same renderer is used by the legacy extraction view and by the
    Patient Record workspace, so the two can never drift apart.  When
    ``conflict_fields`` is supplied, affected fields are flagged inline.
    """
    if data is None:
        st.warning("No clinical data was extracted.")
        return

    conflict_fields = set(conflict_fields or ())

    render_section("Patient Information")
    render_metric_grid(
        [
            {"label": "Patient Name", "value": data.get("patient_name")},
            {"label": "Age", "value": _age_text(data.get("age"))},
            {"label": "Gender", "value": data.get("gender")},
        ]
    )
    _render_field_conflicts(("patient_name", "age", "gender"), conflict_fields)

    render_section("Diagnosis & Cancer Stage")
    render_text_blocks([{"label": "Diagnosis", "value": data.get("diagnosis")}])
    render_metric_grid(
        [
            {"label": "Cancer Type", "value": data.get("cancer_type")},
            {"label": "Cancer Stage", "value": data.get("cancer_stage")},
        ]
    )
    _render_field_conflicts(
        ("diagnosis", "cancer_type", "cancer_stage"), conflict_fields
    )

    render_section(
        "Biomarkers & Test Results",
        "Values are shown exactly as documented; no interpretation is added.",
    )
    render_biomarkers(data.get("biomarkers") or [])
    _render_field_conflicts(("biomarkers",), conflict_fields)

    render_section("Current Medication")
    render_text_blocks(
        [{"label": "Current Medication", "value": data.get("current_medication")}]
    )
    _render_field_conflicts(("current_medication",), conflict_fields)

    render_section("Previous Treatment")
    render_text_blocks(
        [{"label": "Previous Treatment", "value": data.get("previous_treatment")}]
    )
    _render_field_conflicts(("previous_treatment",), conflict_fields)

    render_section("ECOG Performance Status")
    render_metric_grid(
        [{"label": "ECOG Score", "value": data.get("ecog_score")}]
    )
    _render_field_conflicts(("ecog_score",), conflict_fields)

    render_section("Follow-up Plan & Next Steps")
    render_text_blocks(
        [
            {"label": "Follow-up Plan", "value": data.get("follow_up_plan")},
            {"label": "Next Steps", "value": data.get("next_steps")},
        ]
    )
    _render_field_conflicts(("follow_up_plan", "next_steps"), conflict_fields)

    render_section("Adverse Events")
    render_text_blocks([{"label": "Adverse Events", "value": data.get("adverse_events")}])
    _render_field_conflicts(("adverse_events",), conflict_fields)

    if show_export:
        _render_raw_json_export(data)


def _render_raw_json_export(data):
    """Show the raw JSON export block for the legacy extraction view."""
    render_section("Export")
    with st.expander("View raw JSON"):
        st.code(json.dumps(data, indent=2, ensure_ascii=False), language="json")
