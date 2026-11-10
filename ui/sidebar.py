"""
sidebar.py - Sidebar navigation, API status, and patient-case controls.
"""

import streamlit as st

from backend.rag_indexer import generate_patient_id
from ui.session import resolve_api_key
from ui.styles import _gradient_divider


def create_sidebar():
    """Render the sidebar: branding, API status, and patient-case controls."""
    st.sidebar.markdown(
        '<div style="text-align:center; padding: 1rem 0 0.5rem 0;">'
        '<span style="font-size:2.2rem; font-weight:800; background:linear-gradient(135deg,#0f2b4a,#2d9cdb); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;">🏥 OncoLink</span>'
        '<br><span style="font-size:0.8rem; color:#64748b; font-weight:400;">AI Clinical Intelligence Platform</span>'
        '</div>',
        unsafe_allow_html=True
    )
    _gradient_divider("0.5rem 0 1rem 0", st.sidebar)

    st.sidebar.markdown(
        '<div style="font-size:0.85rem; font-weight:700; color:#0f2b4a; text-transform:uppercase; letter-spacing:0.5px; margin:1rem 0 0.5rem 0;">🔑 API Status</div>',
        unsafe_allow_html=True
    )

    # Priority: st.secrets (Streamlit Cloud) > os.getenv (local .env) > sidebar input.
    api_key = resolve_api_key(render_input=True)

    if api_key:
        st.sidebar.markdown(
            '<div style="background:#f0fdf4; border:1px solid #bbf7d0; border-radius:8px; padding:0.5rem 0.8rem;">'
            '<span style="color:#16a34a; font-weight:600; font-size:0.85rem;">✅ NVIDIA API connected</span>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.sidebar.markdown(
            '<div style="background:#fef2f2; border:1px solid #fecaca; border-radius:8px; padding:0.5rem 0.8rem;">'
            '<span style="color:#dc2626; font-weight:600; font-size:0.85rem;">❌ API key not configured</span>'
            '<br><span style="color:#94a3b8; font-size:0.75rem;">Enter your key above to enable AI features.</span>'
            '</div>',
            unsafe_allow_html=True
        )
    
    # --- Patient / case isolation ---
    _gradient_divider("1rem 0 1rem 0", st.sidebar)
    st.sidebar.markdown(
        '<div style="font-size:0.85rem; font-weight:700; color:#0f2b4a; text-transform:uppercase; letter-spacing:0.5px; margin:0 0 0.5rem 0;">🗂 Patient Case</div>',
        unsafe_allow_html=True
    )
    st.sidebar.code(st.session_state.get("patient_id", "case_unknown"), language=None)
    st.sidebar.caption(
        "A random, non-identifying case ID. RAG queries are filtered to this "
        "case only. Start a new case before uploading another patient's records."
    )
    if st.sidebar.button("🆕 Start new case", use_container_width=True):
        st.session_state["patient_id"] = generate_patient_id()
        st.session_state["rag_chat"] = []
        st.session_state["rag_index_result"] = None
        st.session_state["rag_index_signature"] = None
        st.rerun()

    # --- Version Info at the bottom ---
    _gradient_divider("1.5rem 0 0.5rem 0", st.sidebar)
    st.sidebar.markdown(
        '<div style="text-align:center; font-size:0.75rem; color:#94a3b8;">v3.0.0 | Jul 2026</div>',
        unsafe_allow_html=True
    )
