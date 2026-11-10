"""
patient_record.py - The structured Patient Record workspace.

This view reads the extraction and harmonization results already stored in
session state by the application services.  It performs no extraction,
merging, or conflict logic of its own; it only arranges the results so that
missing values, provenance, and disagreements stay visible.
"""

from __future__ import annotations

import streamlit as st

from domain.clinical import FIELD_LABELS, REQUIRED_FIELDS, get_field_label
from ui.components import (
    display_value,
    render_conflicts,
    render_empty_state,
    render_pills,
    render_provenance,
    render_section,
    render_source_documents,
    render_view_header,
)
from ui.extraction import (
    display_structured_clinical_data,
    render_extraction_controls,
)
from ui.harmonization import (
    render_record_exports,
    render_summary,
    run_harmonization,
)


def render_patient_record():
    """Render the structured record for the active patient case."""
    render_view_header(
        "Patient Record",
        "Structured clinical intelligence extracted from the same documents used "
        "by AI Clinical Chat. Missing information and document disagreements are "
        "shown explicitly; nothing is inferred to fill a gap.",
    )

    if "processed_files" not in st.session_state:
        render_empty_state(
            "No documents in this case yet",
            "Upload and process clinical documents to generate a structured "
            "patient record.",
            "Use Uploaded Documents to add PDF, DOCX, or TXT files.",
        )
        return

    extraction = st.session_state.get("extraction_result") or {}
    extraction_done = bool(st.session_state.get("extraction_done"))
    harmonized = st.session_state.get("harmonized_data") or {}
    harmonized_ok = bool(harmonized.get("success"))

    _render_record_summary(extraction, harmonized, harmonized_ok)

    if not (extraction_done and extraction.get("success")):
        # Running extraction updates session state; rerun so the record is
        # rendered once from the stored result instead of twice in one pass.
        if render_extraction_controls(show_heading=True) is not None:
            st.rerun()

        extraction = st.session_state.get("extraction_result") or {}
        extraction_done = bool(st.session_state.get("extraction_done"))
        if not (extraction_done and extraction.get("success")):
            _render_pending_state(extraction, extraction_done)
            return

    if not harmonized_ok:
        _render_harmonize_callout()
        harmonized = st.session_state.get("harmonized_data") or {}
        harmonized_ok = bool(harmonized.get("success"))

    data = harmonized.get("data") if harmonized_ok else extraction.get("data")
    if not data:
        render_empty_state(
            "The record is empty",
            "Extraction completed without usable clinical fields. Review the "
            "extraction messages above.",
        )
        return

    _render_source_banner(harmonized, harmonized_ok)
    display_structured_clinical_data(
        data,
        show_export=False,
        conflict_fields=_conflict_fields(),
    )
    _render_provenance(processed_records(), harmonized, harmonized_ok)
    _render_conflicts_and_warnings(data, extraction, harmonized, harmonized_ok)
    _render_summary_and_exports(data)
    _render_record_workflow(harmonized_ok)


def processed_records():
    """Return the canonical display records for the active case."""
    return list(st.session_state.get("processed_files") or [])


def _render_record_summary(extraction, harmonized, harmonized_ok):
    """Render the record source and compact counts above the sections."""
    documents = [
        record for record in processed_records() if record.get("success")
    ]
    pills = []

    if harmonized_ok:
        pills.append(
            (
                f"Harmonized from {harmonized.get('documents_merged', 0)} document(s)",
                "ok",
            )
        )
        conflicts = (st.session_state.get("conflicts") or {}).get("conflicts", [])
        if conflicts:
            pills.append((f"{len(conflicts)} conflict(s) to review", "warn"))
        else:
            pills.append(("No conflicts detected", "neutral"))
    elif extraction.get("success"):
        pills.append(("Combined extraction", "info"))
        pills.append(("Not yet harmonized", "neutral"))
    else:
        pills.append(("Extraction pending", "neutral"))

    if documents:
        pills.append((f"{len(documents)} source document(s)", "neutral"))

    render_pills(pills)


def _render_pending_state(extraction, extraction_done):
    """Explain why no structured record is available yet."""
    if extraction_done and not extraction.get("success"):
        render_empty_state(
            "Extraction did not produce a record",
            "The model call or validation step failed, so no clinical values were "
            "stored. The error is shown below; retrying does not erase your "
            "uploaded documents.",
        )
        for error in extraction.get("errors", []):
            st.error(error)
        return

    if extraction.get("errors"):
        for error in extraction.get("errors", []):
            st.error(error)

    render_empty_state(
        "No structured record yet",
        "Run AI extraction to turn the uploaded documents into a structured "
        "patient record. Missing values are reported explicitly rather than "
        "inferred.",
        "Extraction requires an NVIDIA API key configured in the sidebar.",
    )


def _render_harmonize_callout():
    """Offer harmonization while still showing the combined extraction."""
    st.info(
        "This record comes from the combined extraction pass. Run harmonization "
        "to merge the per-document records, record field provenance, and compare "
        "the documents for conflicts."
    )
    if st.button(
        "Run harmonization",
        type="primary",
        key="record_harmonize_button",
    ):
        run_harmonization()


def _render_source_banner(harmonized, harmonized_ok):
    """State which record is being displayed."""
    if harmonized_ok:
        render_section(
            "Harmonized Record",
            "Merged values from the per-document extractions. Conflicts are "
            "reported below rather than resolved silently.",
        )
    else:
        render_section(
            "Combined Extraction",
            "A single extraction across all uploaded documents. Field-level "
            "provenance and conflict detection become available after "
            "harmonization.",
        )


def _provenance_rows(harmonized):
    """Build ordered provenance rows from the harmonizer's field sources."""
    field_sources = harmonized.get("field_sources") or {}
    if not isinstance(field_sources, dict):
        return []

    ordered_fields = list(FIELD_LABELS)
    ordered_fields.extend(
        field for field in field_sources if field not in ordered_fields
    )

    rows = []
    for field in ordered_fields:
        if field not in field_sources:
            continue
        sources = field_sources.get(field)
        if isinstance(sources, (list, tuple, set)):
            sources = "; ".join(str(source) for source in sources)
        rows.append(
            {
                "label": get_field_label(field),
                "sources": sources or "Not recorded",
            }
        )
    return rows


def _render_provenance(records, harmonized, harmonized_ok):
    """Render document sources and, when available, field-level provenance."""
    render_section(
        "Document Sources & Provenance",
        "Documents that fed this record, plus the source of each merged field.",
    )

    if records:
        with st.expander(
            f"Source documents ({len(records)})", expanded=not harmonized_ok
        ):
            render_source_documents(records)
    else:
        render_source_documents(records)

    if harmonized_ok:
        rows = _provenance_rows(harmonized)
        if rows:
            render_provenance(rows)
        else:
            render_empty_state(
                "No field-level provenance was recorded",
                "The harmonized result did not include a field-to-document map.",
            )
    else:
        st.caption(
            "Field-level provenance is recorded when the per-document records are "
            "harmonized."
        )


def _conflict_fields():
    """Return the set of fields the conflict detector flagged."""
    conflicts = (st.session_state.get("conflicts") or {}).get("conflicts", [])
    return {
        conflict.get("field")
        for conflict in conflicts
        if isinstance(conflict, dict) and conflict.get("field")
    }


def _missing_required_fields(data):
    """Return labels for expected fields that the documents did not support."""
    missing = []
    for field in REQUIRED_FIELDS:
        _, is_missing = display_value(data.get(field))
        if is_missing:
            missing.append(get_field_label(field))
    return missing


def _collect_warnings(extraction, harmonized):
    """Collect extraction and harmonization warnings without duplicates."""
    warnings = []
    for source in (extraction, harmonized):
        for warning in source.get("warnings") or []:
            if warning not in warnings:
                warnings.append(warning)
    return warnings


def _render_conflicts_and_warnings(data, extraction, harmonized, harmonized_ok):
    """Render conflicts, warnings, and explicitly missing expected fields."""
    render_section(
        "Conflicts & Warnings",
        "Disagreements between documents and information the documents did not "
        "contain.",
    )

    missing = _missing_required_fields(data)
    if missing:
        st.warning(
            "Expected fields not documented: " + ", ".join(missing) + ". "
            "These values were left empty, not inferred."
        )
    else:
        st.caption("All expected fields are present in the displayed record.")

    warnings = _collect_warnings(extraction, harmonized)
    for warning in warnings:
        st.warning(warning)

    if harmonized_ok:
        conflicts = (st.session_state.get("conflicts") or {}).get("conflicts", [])
        render_conflicts(conflicts)
    else:
        st.info(
            "Conflicts are detected during harmonization, when each document's "
            "record is compared with the others."
        )


def _render_summary_and_exports(data):
    """Render the existing summary and record exports."""
    summary = st.session_state.get("summary") or {}
    if summary.get("success") and summary.get("summary"):
        with st.expander("Generated patient summary", expanded=False):
            render_summary(summary)

    render_record_exports(data)


def _render_record_workflow(harmonized_ok):
    """Expose re-run controls without cluttering the record itself."""
    with st.expander("Record workflow", expanded=False):
        st.caption(
            "Re-running extraction replaces the current record and clears the "
            "previous harmonization. Uploaded documents are not changed."
        )
        result = render_extraction_controls(
            show_heading=False,
            label="Re-run AI extraction",
            key="record_reextract_button",
        )
        if result is not None:
            st.rerun()

        if harmonized_ok:
            if st.button(
                "Re-run harmonization",
                key="record_reharmonize_button",
            ):
                run_harmonization()
                st.rerun()
