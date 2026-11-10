"""
session.py - Logging, per-session workflow state, and API-key resolution.

Streamlit re-runs the whole script on every interaction, so anything that
must survive a rerun lives in ``st.session_state``.  This module centralizes
that setup, the API-key lookup order, and the cached document-intelligence
service used by the UI.
"""

import logging
import os

import streamlit as st

from application.intelligence import DocumentIntelligenceService
from domain.documents import generate_patient_id
from infrastructure.config import RAGConfig
from infrastructure.logging import setup_logging as _setup_file_logging

# Placeholder shipped in the example secrets file; it must not be treated as
# a real key.
_PLACEHOLDER_API_KEY = "nvapi-your-key-here"
_startup_logged = False


def setup_logging() -> None:
    """Configure rotating application logging once per process."""
    global _startup_logged

    _setup_file_logging()
    if not _startup_logged:
        logging.info("ONCOLINK APPLICATION STARTED")
        _startup_logged = True


def initialize_rag_session_state() -> None:
    """
    Create the per-session RAG state.

    The patient/case ID is a random, non-identifying value for this browser
    session.  Every vector-store query filters on it, so one session cannot
    retrieve another session's documents.
    """
    defaults = {
        "patient_id": generate_patient_id,
        "rag_chat": list,
        "intelligence_service": lambda: None,
    }
    for key, factory in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = factory()

    if "rag_settings" not in st.session_state:
        st.session_state["rag_settings"] = _default_rag_settings()


def _default_rag_settings() -> dict:
    """Build UI defaults from the environment-aware RAG configuration."""
    config = RAGConfig()
    threshold = config.score_threshold
    if threshold is None:
        threshold = -1.0
    return {
        "mode": "hybrid" if config.hybrid_search else "semantic",
        "top_k": config.top_k,
        "rerank": config.rerank_enabled,
        "threshold": max(-1.0, min(1.0, float(threshold))),
    }


def get_intelligence_service() -> DocumentIntelligenceService:
    """Return the workflow service for the current patient case.

    The service is cached in session state so canonical documents and lazy RAG
    components survive Streamlit reruns.  If the case ID changes, a fresh
    service is created so one case's data can never leak into another.
    """
    patient_id = st.session_state.get("patient_id") or generate_patient_id()
    st.session_state["patient_id"] = patient_id

    service = st.session_state.get("intelligence_service")
    if service is None or service.patient_id != patient_id:
        service = DocumentIntelligenceService(patient_id=patient_id)
        st.session_state["intelligence_service"] = service
    return service


def reset_case_state() -> DocumentIntelligenceService:
    """Start a new patient case and clear every case-scoped UI value."""
    service = st.session_state.get("intelligence_service")
    if service is None:
        service = DocumentIntelligenceService()

    service.reset()
    st.session_state["patient_id"] = service.patient_id
    st.session_state["intelligence_service"] = service

    for key in (
        "processed_files",
        "processing_done",
        "extraction_result",
        "per_doc_results",
        "extraction_done",
        "harmonized_data",
        "conflicts",
        "summary",
        "harmonization_done",
        "rag_chat",
        "document_uploader",
    ):
        st.session_state.pop(key, None)

    return service


def resolve_api_key(render_input: bool = False):
    """
    Resolve the NVIDIA API key.

    Priority order is preserved: Streamlit secrets, then environment, then the
    sidebar input.  ``render_input=True`` draws the sidebar password field as
    the final fallback.  A resolved key is exported to ``os.environ`` so
    backend modules see the same value.
    """
    api_key = _key_from_secrets() or _clean_api_key(os.getenv("NVIDIA_API_KEY"))
    if api_key:
        os.environ["NVIDIA_API_KEY"] = api_key
        return api_key

    stored_key = _clean_api_key(st.session_state.get("nvidia_api_key_input"))
    if stored_key:
        os.environ["NVIDIA_API_KEY"] = stored_key
        return stored_key

    if not render_input:
        return None

    entered_key = _clean_api_key(
        st.sidebar.text_input(
            "NVIDIA API Key",
            type="password",
            placeholder="nvapi-...",
            help="Get a free key at: https://build.nvidia.com/",
            value=st.session_state.get("nvidia_api_key_input", ""),
        )
    )
    if entered_key:
        st.session_state["nvidia_api_key_input"] = entered_key
        os.environ["NVIDIA_API_KEY"] = entered_key
        return entered_key
    return None


def _key_from_secrets():
    """Read the key from Streamlit secrets when they are available."""
    try:
        if "NVIDIA_API_KEY" in st.secrets:
            return _clean_api_key(st.secrets["NVIDIA_API_KEY"])
    except Exception:
        # Secrets are optional locally; a missing file is not an error.
        return None
    return None


def _clean_api_key(value) -> str | None:
    """Normalize a key and reject blanks or the example placeholder."""
    value = (value or "").strip()
    if not value or value == _PLACEHOLDER_API_KEY:
        return None
    return value
