"""
harmonization.py - Harmonization, conflict, summary, and status UI.

The merging and summary logic lives in ``application.clinical`` /
``backend``; this module triggers it, stores the results, and renders them.
"""

import json

import streamlit as st

from ui.export import _text_to_pdf_bytes
from ui.extraction import display_structured_clinical_data
from ui.session import get_intelligence_service
from ui.styles import _gradient_divider


def show_harmonization_section():
    """Render the harmonization controls and the resulting record."""
    _render_header()

    harmonize_clicked = st.button(
        "🔄 Run Harmonization", type="primary", use_container_width=True
    )
    if harmonize_clicked:
        _run_harmonization()

    harmonized = st.session_state.get("harmonized_data", {})
    summary = st.session_state.get("summary", {})

    if not harmonized.get("success"):
        return

    with st.expander("📊 Harmonized Patient Record", expanded=True):
        display_structured_clinical_data(harmonized["data"], show_export=False)

    _render_summary(summary)
    _render_export(harmonized["data"])


def _render_header():
    """Render the section title and description."""
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:2rem;">'
        '<span style="font-size:1.3rem;">🔗</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">Harmonization & Summary</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">'
        "Merge data from all documents and generate a unified patient summary.</p>",
        unsafe_allow_html=True,
    )


def _run_harmonization():
    """Harmonize the per-document extractions and store all derived results."""
    service = get_intelligence_service()

    with st.spinner("Harmonizing data across documents..."):
        outcome = service.harmonize()

        st.session_state["harmonized_data"] = outcome.harmonized
        st.session_state["conflicts"] = outcome.conflicts
        st.session_state["summary"] = outcome.summary
        st.session_state["harmonization_done"] = True

    if outcome.harmonized.get("success"):
        st.success(
            "✅ Harmonization complete! "
            f"{outcome.harmonized.get('documents_merged', 0)} document(s) merged."
        )


def _render_summary(summary):
    """Render the patient summary with a PDF download (TXT fallback)."""
    if not (summary and summary.get("success") and summary.get("summary")):
        return

    st.markdown(
        '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin:1.5rem 0 0.5rem 0;">📋 Patient Summary</div>',
        unsafe_allow_html=True,
    )
    summary_text = summary["summary"]
    st.code(summary_text, language="text")

    try:
        st.download_button(
            "⬇ PDF",
            data=_text_to_pdf_bytes(summary_text),
            file_name="patient_summary.pdf",
            mime="application/pdf",
            key="dl_ps_pdf",
        )
    except Exception:
        # FPDF can be unavailable on some deployments; plain text always works.
        st.download_button(
            "⬇ PDF",
            data=summary_text,
            file_name="patient_summary.txt",
            mime="text/plain",
            key="dl_ps_fb",
        )


def _render_export(harmonized_data):
    """Render the harmonized-record export and raw JSON view."""
    _gradient_divider("1.5rem 0 1rem 0")
    st.markdown(
        '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin-bottom:0.5rem;">📥 Export Harmonized Record</div>',
        unsafe_allow_html=True,
    )

    harmonized_json = json.dumps(harmonized_data, indent=2)
    try:
        st.download_button(
            "⬇ PDF",
            data=_text_to_pdf_bytes(harmonized_json),
            file_name="harmonized_record.pdf",
            mime="application/pdf",
            key="dl_hr_pdf",
        )
    except Exception:
        st.download_button(
            "⬇ PDF",
            data=harmonized_json,
            file_name="harmonized_record.txt",
            mime="text/plain",
            key="dl_hr_fb",
        )

    with st.expander("View Harmonized JSON"):
        st.code(harmonized_json, language="json")


def create_status_panel(processing_done):
    """Render the conflict and status summary at the bottom of the page."""
    col1, col2 = st.columns(2)
    with col1:
        _render_conflicts_panel(processing_done)
    with col2:
        _render_status_panel(processing_done)


def _render_conflicts_panel(processing_done):
    """Show detected conflicts, or the next action needed to detect them."""
    st.markdown('<div class="section-heading">⚠️ Conflicts</div>', unsafe_allow_html=True)

    if not st.session_state.get("harmonization_done", False):
        _render_pre_harmonization_hint(processing_done)
        return

    conflicts = st.session_state.get("conflicts", {}).get("conflicts", [])
    if not conflicts:
        st.success("✅ No conflicts detected across documents.")
        return

    high_count = sum(1 for conflict in conflicts if conflict["severity"] == "high")
    medium_count = len(conflicts) - high_count
    st.info(
        f"⚠️ {len(conflicts)} conflict(s) found — {high_count} high, {medium_count} medium"
    )

    for conflict in conflicts:
        severity = conflict.get("severity", "medium")
        icon = "🔴" if severity == "high" else "🟡"
        description = conflict.get("description", conflict.get("field", "Unknown"))
        st.markdown(f"{icon} {description}")
        for document, value in conflict.get("sources", []):
            document_name = document.split("\\")[-1].split("/")[-1]
            st.markdown(f"📄 {document_name} → {value}")
        st.markdown("---")


def _render_pre_harmonization_hint(processing_done):
    """Explain which step is needed before conflicts can be shown."""
    if st.session_state.get("extraction_done", False):
        st.info('🔍 Run "Harmonization" to detect data conflicts.')
    elif processing_done:
        st.info("🔄 Run AI extraction to identify potential issues.")
    else:
        st.info("✅ No warnings.")


def _render_status_panel(processing_done):
    """Show how far the current case has progressed."""
    st.markdown('<div class="section-heading">📊 Status</div>', unsafe_allow_html=True)

    harmonization_done = st.session_state.get("harmonization_done", False)
    harmonized = st.session_state.get("harmonized_data", {})
    extraction_done = st.session_state.get("extraction_done", False)

    if harmonization_done and harmonized.get("success"):
        st.success(f"✅ {harmonized['documents_merged']} document(s) harmonized.")
    elif extraction_done:
        st.info('✅ Structured data extracted. Run "Harmonization" to merge.')
    elif processing_done:
        st.info('⏳ Click "Run AI Extraction" to extract structured data.')
    else:
        st.info("⏸️ Waiting for analysis.")
