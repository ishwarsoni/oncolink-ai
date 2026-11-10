"""
harmonization.py - Harmonization, conflict, summary, export, and status UI.

The merging and summary logic lives in ``application.clinical`` /
``backend``; this module triggers it, stores the results, and renders them.
"""

import json

import streamlit as st

from ui.components import (
    render_conflicts,
    render_pills,
    render_section,
    render_text_blocks,
)
from ui.export import _text_to_pdf_bytes
from ui.extraction import display_structured_clinical_data
from ui.session import get_intelligence_service


def run_harmonization():
    """Harmonize per-document extractions and persist every derived result."""
    service = get_intelligence_service()

    with st.spinner("Harmonizing data across documents..."):
        outcome = service.harmonize()

        st.session_state["harmonized_data"] = outcome.harmonized
        st.session_state["conflicts"] = outcome.conflicts
        st.session_state["summary"] = outcome.summary
        st.session_state["harmonization_done"] = True

    if outcome.harmonized.get("success"):
        st.success(
            "Harmonization complete. "
            f"{outcome.harmonized.get('documents_merged', 0)} document(s) merged."
        )
    else:
        warnings = outcome.harmonized.get("warnings") or []
        for warning in warnings:
            st.warning(warning)
        st.error(
            "Harmonization did not produce a merged record. Review the per-document "
            "extraction results first."
        )
    return outcome


def show_harmonization_section():
    """Render the harmonization controls and the resulting record."""
    render_section(
        "Harmonization & Summary",
        "Merge the per-document extractions, review disagreements, and generate "
        "a summary from the merged record.",
    )

    harmonize_clicked = st.button(
        "Run harmonization",
        type="primary",
        key="run_harmonization_button",
    )
    if harmonize_clicked:
        run_harmonization()

    harmonized = st.session_state.get("harmonized_data") or {}
    summary = st.session_state.get("summary") or {}

    if not harmonized.get("success"):
        return

    with st.expander("Harmonized patient record", expanded=True):
        display_structured_clinical_data(harmonized["data"], show_export=False)

    render_summary(summary)
    render_record_exports(harmonized["data"])


def render_summary(summary):
    """Render the generated patient summary with a PDF download (TXT fallback)."""
    if not (summary and summary.get("success") and summary.get("summary")):
        return

    render_section(
        "Generated Patient Summary",
        "Narrative summary generated from the harmonized record.",
    )
    summary_text = summary["summary"]
    render_text_blocks([{"label": "Summary", "value": summary_text}])

    try:
        st.download_button(
            "Download summary PDF",
            data=_text_to_pdf_bytes(summary_text),
            file_name="patient_summary.pdf",
            mime="application/pdf",
            key="dl_ps_pdf",
        )
    except Exception:
        # FPDF can be unavailable on some deployments; plain text always works.
        st.download_button(
            "Download summary (TXT)",
            data=summary_text,
            file_name="patient_summary.txt",
            mime="text/plain",
            key="dl_ps_fb",
        )


def render_record_exports(data):
    """Render the existing PDF and JSON exports for a clinical record."""
    if not data:
        return

    render_section(
        "Export",
        "Downloads contain the record currently displayed in this workspace.",
    )
    record_json = json.dumps(data, indent=2, ensure_ascii=False)

    pdf_col, json_col = st.columns(2)
    with pdf_col:
        try:
            st.download_button(
                "Download PDF",
                data=_text_to_pdf_bytes(record_json),
                file_name="harmonized_record.pdf",
                mime="application/pdf",
                key="record_pdf",
                use_container_width=True,
            )
        except Exception:
            st.download_button(
                "Download PDF",
                data=record_json,
                file_name="harmonized_record.txt",
                mime="text/plain",
                key="record_pdf_fallback",
                use_container_width=True,
            )
    with json_col:
        st.download_button(
            "Download JSON",
            data=record_json,
            file_name="harmonized_record.json",
            mime="application/json",
            key="record_json",
            use_container_width=True,
        )
    st.caption(
        "The PDF is a plain-text rendering. The JSON carries the same fields as "
        "the clinical record, without interpretation."
    )


def create_status_panel(processing_done):
    """Render the conflict and status summary at the bottom of the page."""
    col1, col2 = st.columns(2)
    with col1:
        _render_conflicts_panel(processing_done)
    with col2:
        _render_status_panel(processing_done)


def _render_conflicts_panel(processing_done):
    """Show detected conflicts, or the next action needed to detect them."""
    render_section("Conflicts")

    conflicts_result = st.session_state.get("conflicts") or {}
    if not st.session_state.get("harmonization_done", False):
        _render_pre_harmonization_hint(processing_done)
        return

    conflicts = conflicts_result.get("conflicts", [])
    render_conflicts(conflicts)


def _render_pre_harmonization_hint(processing_done):
    """Explain which step is needed before conflicts can be shown."""
    if st.session_state.get("extraction_done", False):
        st.info('Run harmonization to compare documents and detect conflicts.')
    elif processing_done:
        st.info("Run AI extraction to identify potential issues.")
    else:
        st.info("No warnings.")


def _render_status_panel(processing_done):
    """Show how far the current case has progressed."""
    render_section("Status")

    harmonization_done = st.session_state.get("harmonization_done", False)
    harmonized = st.session_state.get("harmonized_data") or {}
    extraction_done = st.session_state.get("extraction_done", False)

    if harmonization_done and harmonized.get("success"):
        render_pills(
            [
                (
                    f"{harmonized.get('documents_merged', 0)} document(s) harmonized",
                    "ok",
                )
            ]
        )
    elif extraction_done:
        st.info("Structured data extracted. Run harmonization to merge it.")
    elif processing_done:
        st.info("Click Run AI extraction to extract structured data.")
    else:
        st.info("Waiting for analysis.")
