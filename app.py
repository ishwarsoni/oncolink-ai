"""
OncoLink - AI Clinical Intelligence Platform.

Application entry point.  The Streamlit UI lives in the ``ui`` package and
all business logic lives in ``backend``; this file only wires them together
in the intended render order.

Run with:
    streamlit run app.py
"""

import streamlit as st
from dotenv import load_dotenv

from ui.extraction import create_extraction_section
from ui.harmonization import create_status_panel, show_harmonization_section
from ui.rag import create_rag_assistant_section
from ui.session import initialize_rag_session_state, setup_logging
from ui.sidebar import create_sidebar
from ui.styles import configure_page, create_footer, load_custom_css
from ui.upload import (
    create_file_uploader,
    create_process_button,
    display_extracted_text,
    display_uploaded_files,
)


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
    st.divider()

    uploaded_files = create_file_uploader()
    st.markdown("<br>", unsafe_allow_html=True)
    display_uploaded_files(uploaded_files)

    st.markdown("<br>", unsafe_allow_html=True)
    processing_done = create_process_button(uploaded_files)
    display_extracted_text()

    create_extraction_section()

    # Harmonization is only meaningful after a successful extraction.
    if st.session_state.get("extraction_done"):
        show_harmonization_section()

    create_rag_assistant_section()

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    create_status_panel(processing_done)
    create_footer()


def _render_page_header():
    """Render the logo, title, subtitle, and one-line description."""
    col_logo, col_text = st.columns([0.08, 0.92])
    with col_logo:
        st.markdown(
            '<div style="font-size:2.5rem; line-height:1;">🏥</div>',
            unsafe_allow_html=True,
        )
    with col_text:
        st.markdown('<div class="main-title">OncoLink</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="main-subtitle">AI Clinical Intelligence Platform</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="main-description">Upload oncology reports — referral letters, '
        "pathology reports, lab reports, and EMR notes — and generate structured, "
        "unified patient summaries.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
