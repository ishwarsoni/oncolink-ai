"""
sidebar.py - Sidebar navigation, API status, and patient-case controls.
"""

import streamlit as st

from ui.components import esc, render_pills
from ui.session import reset_case_state, resolve_api_key
from ui.styles import _gradient_divider


def create_sidebar():
    """Render the sidebar: branding, API status, and patient-case controls."""
    st.sidebar.markdown(
        '<div class="ol-brand">'
        '<div class="ol-brand-name">OncoLink</div>'
        '<div class="ol-brand-sub">Clinical intelligence workspace</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    _gradient_divider("0.7rem 0 0.9rem 0", st.sidebar)

    st.sidebar.markdown(
        '<div class="ol-sidebar-label">NVIDIA API</div>',
        unsafe_allow_html=True,
    )

    # Priority: st.secrets (Streamlit Cloud) > os.getenv (local .env) > sidebar input.
    api_key = resolve_api_key(render_input=True)

    if api_key:
        render_pills([("Connected", "ok")], container=st.sidebar)
        st.sidebar.caption(
            "Extraction and answer generation use the configured Nemotron model."
        )
    else:
        render_pills([("Not configured", "bad")], container=st.sidebar)
        st.sidebar.caption(
            "Enter a key above to enable extraction and answer generation. "
            "Document text and indexing remain local."
        )

    # --- Patient / case isolation ---
    _gradient_divider("1rem 0 1rem 0", st.sidebar)
    st.sidebar.markdown(
        '<div class="ol-sidebar-label">Patient case</div>',
        unsafe_allow_html=True,
    )
    patient_id = st.session_state.get("patient_id", "case_unknown")
    st.sidebar.markdown(
        f'<div class="ol-file-row"><div class="ol-file-name">{esc(patient_id)}</div>'
        '<span class="ol-pill ol-pill--neutral">Active</span></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "A random, non-identifying case ID. Retrieval is filtered to this case "
        "only. Start a new case before uploading another patient's records."
    )
    if st.sidebar.button(
        "Start new case",
        key="new_case_button",
        use_container_width=True,
    ):
        reset_case_state()
        st.rerun()

    # --- Version info at the bottom ---
    _gradient_divider("1.5rem 0 0.5rem 0", st.sidebar)
    st.sidebar.markdown(
        '<div class="ol-sidebar-note" style="text-align:center;">'
        "OncoLink v3.0 · Educational use only"
        "</div>",
        unsafe_allow_html=True,
    )
