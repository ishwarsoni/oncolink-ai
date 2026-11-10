"""
extraction.py - AI clinical extraction UI and structured-data display.

The extraction pipeline itself lives in ``backend.extractor``.  This module
only starts it, stores the results in session state, and renders them.
"""

import json

import streamlit as st

from backend.extractor import extract_each_document, extract_from_multiple_documents
from ui.session import resolve_api_key
from ui.styles import _gradient_divider


def create_extraction_section():
    """Render the extraction button, feedback, and extracted data."""
    if "processed_files" not in st.session_state:
        return

    st.divider()
    _render_section_header()
    api_ready = _render_api_warning()

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        extraction_clicked = st.button(
            "🧠 Run AI Extraction",
            type="primary",
            use_container_width=True,
            disabled=not api_ready,
        )
        if extraction_clicked:
            _run_and_store_extraction()

    st.markdown("<br>", unsafe_allow_html=True)
    _render_extraction_results()
    st.markdown("<br>", unsafe_allow_html=True)


def _render_section_header():
    """Render the section title and description."""
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-top:1.5rem;">'
        '<span style="font-size:1.3rem;">🤖</span>'
        '<span style="font-size:1.2rem; font-weight:700; color:#0f2b4a;">AI Clinical Extraction</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#64748b; font-size:0.9rem; margin-top:-0.3rem;">'
        "Extract structured clinical information from all documents using NVIDIA's Nemotron-3 Ultra AI model."
        "</p>",
        unsafe_allow_html=True,
    )


def _render_api_warning() -> bool:
    """Warn when no API key is configured and return whether one exists."""
    api_ready = resolve_api_key() is not None
    if not api_ready:
        st.warning(
            "⚠️ NVIDIA API key not found.\n\n"
            "To enable AI extraction, please **enter your NVIDIA API Key in the sidebar input field**.\n\n"
            "Get a free API key at: https://build.nvidia.com/"
        )
    return api_ready


def _run_and_store_extraction():
    """
    Run combined and per-document extraction, then persist the results.

    The combined result powers the main extraction view; the per-document
    results feed harmonization and conflict detection.
    """
    processed_files = st.session_state["processed_files"]

    with st.spinner("Calling NVIDIA AI... This may take 10-30 seconds."):
        result = extract_from_multiple_documents(processed_files)
        per_document_results = extract_each_document(processed_files)

        st.session_state["extraction_result"] = result
        st.session_state["per_doc_results"] = per_document_results
        st.session_state["extraction_done"] = True
        st.session_state["harmonization_done"] = False

    _render_extraction_feedback(result)


def _render_extraction_feedback(result):
    """Show success, validation warnings, or the errors from a run."""
    if result["success"]:
        st.success("✅ AI extraction completed successfully!")
        for warning in result.get("warnings", []):
            st.warning(f"⚠️ {warning}")
        return

    errors = result.get("errors", [])
    if errors:
        st.error("❌ Extraction failed:\n" + "\n".join(errors))
    else:
        st.error("❌ Extraction failed with an unknown error.")


def _render_extraction_results():
    """Render the most recent extraction result, if there is one."""
    if "extraction_result" not in st.session_state:
        return

    result = st.session_state["extraction_result"]
    if result["success"]:
        validation = result.get("validation", {})
        if validation and validation.get("valid"):
            st.success("✅ Data validated against Pydantic schema")

        normalization = result.get("normalization", {})
        if normalization and normalization.get("success"):
            st.info("🔄 Values normalized for consistency")

        display_structured_clinical_data(result["data"])
        return

    for error in result.get("errors", []):
        st.error(f"❌ {error}")
    if result.get("raw_response"):
        with st.expander("🔧 Debug: Raw LLM Response"):
            st.text(result["raw_response"])


def _render_section_heading(icon, title):
    """Render one structured-data section heading."""
    st.markdown(
        '<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.8rem;">'
        f'<span style="font-size:1.1rem;">{icon}</span>'
        f'<span style="font-size:1.05rem; font-weight:700; color:#0f2b4a;">{title}</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def _render_metric_card(label, value):
    """Render one small centered value card."""
    st.markdown(
        '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; '
        'padding:0.8rem 1rem; text-align:center;">'
        f'<div style="font-size:0.7rem; font-weight:600; color:#64748b; '
        f'text-transform:uppercase; letter-spacing:0.5px;">{label}</div>'
        f'<div style="font-size:0.9rem; font-weight:600; color:#0f2b4a; margin-top:0.2rem;">'
        f"{value}</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _render_patient_information(data):
    """Show the patient name, age, and gender cards."""
    _render_section_heading("👤", "Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        value = data.get("patient_name")
        st.metric(label="Patient Name", value=value if value else "Not found")

    with col2:
        value = data.get("age")
        st.metric(
            label="Age", value=f"{value} years" if value is not None else "Not found"
        )

    with col3:
        value = data.get("gender")
        st.metric(label="Gender", value=value if value else "Not found")

    st.markdown("---")


def _render_cancer_information(data):
    """Show diagnosis, cancer type, stage, and ECOG score."""
    _render_section_heading("🧬", "Cancer Information")

    diagnosis = data.get("diagnosis")
    st.markdown(
        '<div style="background:#ffffff; border:1px solid #e2e8f0; border-radius:10px; '
        'padding:1rem 1.2rem; margin-bottom:0.8rem;">'
        '<div style="font-size:0.75rem; font-weight:600; color:#64748b; '
        'text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.3rem;">Diagnosis</div>'
        f'<div style="font-size:0.95rem; color:#0f2b4a; font-weight:500;">'
        f'{diagnosis if diagnosis else "Not found"}</div>'
        "</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        _render_metric_card("Cancer Type", data.get("cancer_type") or "Not found")
    with col2:
        _render_metric_card("Cancer Stage", data.get("cancer_stage") or "Not found")
    with col3:
        ecog_score = data.get("ecog_score")
        _render_metric_card(
            "ECOG Score", str(ecog_score) if ecog_score is not None else "Not found"
        )

    st.markdown("---")


def _render_biomarkers(data):
    """Show each extracted biomarker in its own column."""
    _render_section_heading("🔬", "Biomarkers")

    biomarkers = data.get("biomarkers")
    if not biomarkers:
        st.info("No biomarker information found in the documents.")
        st.markdown("---")
        return

    biomarker_columns = st.columns(len(biomarkers))
    for index, biomarker in enumerate(biomarkers):
        with biomarker_columns[index]:
            st.markdown(f"**{biomarker.get('name', 'Unknown')}**")
            st.markdown(f"Value: {biomarker.get('value', 'N/A')}")
            st.markdown(f"Status: {biomarker.get('status', 'N/A')}")

    st.markdown("---")


def _render_treatment_information(data):
    """Show current medication and previous treatment."""
    _render_section_heading("💊", "Treatment Information")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Current Medication**")
        st.info(data.get("current_medication") or "Not found")
    with col2:
        st.markdown("**Previous Treatment**")
        st.info(data.get("previous_treatment") or "Not found")

    st.markdown("---")


def _render_plan(data):
    """Show the follow-up plan and next steps."""
    _render_section_heading("📋", "Plan")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Follow-up Plan**")
        st.info(data.get("follow_up_plan") or "Not found")
    with col2:
        st.markdown("**Next Steps**")
        st.info(data.get("next_steps") or "Not found")


def _render_raw_json_export(data):
    """Show the raw JSON export block."""
    _gradient_divider("1.5rem 0 1rem 0")
    st.markdown(
        '<div style="font-size:0.95rem; font-weight:700; color:#0f2b4a; margin-bottom:0.5rem;">📥 Export Data</div>',
        unsafe_allow_html=True,
    )

    with st.expander("View Raw JSON"):
        st.code(json.dumps(data, indent=2), language="json")


def display_structured_clinical_data(data, show_export=True):
    """
    Display extracted clinical data in professional cards.

    Section order is preserved from the original layout: patient, cancer,
    biomarkers, treatment, plan, then the JSON export.
    """
    if data is None:
        st.warning("No clinical data was extracted.")
        return

    _render_patient_information(data)
    _render_cancer_information(data)
    _render_biomarkers(data)
    _render_treatment_information(data)
    _render_plan(data)

    if show_export:
        _render_raw_json_export(data)
