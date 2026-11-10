"""
OncoLink - AI Clinical Intelligence Platform.

Application entry point.  The Streamlit UI lives in the ``ui`` package,
workflows live in ``application``, domain models in ``domain``, clinical and
retrieval algorithms in ``backend``, and external services in
``infrastructure``.  This file only wires them together in render order.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from ui.components import render_product_header, render_status_strip
from ui.patient_record import render_patient_record
from ui.rag import create_rag_assistant_section
from ui.session import (
    get_intelligence_service,
    initialize_rag_session_state,
    setup_logging,
)
from ui.sidebar import create_sidebar
from ui.styles import configure_page, create_footer, load_custom_css
from ui.upload import (
    render_documents_intake,
    render_documents_view,
    render_processing_notice,
)

WORKSPACE_VIEWS = ("Patient Record", "AI Clinical Chat", "Uploaded Documents")
DEFAULT_VIEW = WORKSPACE_VIEWS[0]


def main():
    """Render the full application."""
    # Environment variables must be loaded before any config object is built.
    load_dotenv()
    setup_logging()

    # configure_page must be the first Streamlit command in the script.
    configure_page()
    initialize_rag_session_state()
    load_custom_css()
    create_sidebar()

    _render_page_header()

    if not st.session_state.get("processed_files"):
        render_processing_notice()
        render_documents_intake()
        _render_case_status()
        create_footer()
        return

    view = _render_workspace_navigation()
    render_processing_notice()

    if view == "AI Clinical Chat":
        create_rag_assistant_section()
    elif view == "Uploaded Documents":
        render_documents_view()
    else:
        render_patient_record()

    _render_case_status()
    create_footer()


def _render_page_header():
    """Render the compact product banner."""
    render_product_header(
        title="OncoLink",
        subtitle="Clinical intelligence workspace.",
        description=(
            "Turn uploaded oncology documents into a structured patient record "
            "and ask evidence-grounded questions about the same document set."
        ),
        eyebrow="OncoLink · AI Clinical Intelligence",
    )


def _render_workspace_navigation():
    """Render the workspace switcher and return the active view name."""
    current = st.session_state.get("workspace_view") or DEFAULT_VIEW
    if current not in WORKSPACE_VIEWS:
        current = DEFAULT_VIEW
    st.session_state["workspace_view"] = current

    selected = st.radio(
        "Workspace view",
        WORKSPACE_VIEWS,
        key="workspace_view",
        horizontal=True,
        label_visibility="collapsed",
    )
    if selected not in WORKSPACE_VIEWS:
        selected = DEFAULT_VIEW
    return selected


def _render_case_status():
    """Render the independent ingestion, clinical, and RAG stage states."""
    service = get_intelligence_service()
    stages = [status.as_dict() for status in service.statuses.values()]
    render_status_strip(stages)


if __name__ == "__main__":
    main()
