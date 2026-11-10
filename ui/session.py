"""
session.py - Logging, per-session RAG state, and API-key resolution.

Streamlit re-runs the whole script on every interaction, so anything that
must survive a rerun lives in ``st.session_state``.  This module centralizes
that setup and the API-key lookup order used by the rest of the UI.
"""

import logging
import os
from pathlib import Path

import streamlit as st

from backend.config import RAGConfig
from backend.rag_indexer import generate_patient_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "application.log"

# Placeholder shipped in the example secrets file; it must not be treated as
# a real key.
_PLACEHOLDER_API_KEY = "nvapi-your-key-here"


def setup_logging() -> None:
    """Configure file logging for the application."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=str(LOG_FILE),
        # Start each run with a clean log; set to "a" to keep history.
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logging.info("=" * 60)
    logging.info("ONCOLINK APPLICATION STARTED")
    logging.info("=" * 60)


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
        "rag_components": lambda: None,
        "rag_index_result": lambda: None,
        "rag_index_signature": lambda: None,
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
