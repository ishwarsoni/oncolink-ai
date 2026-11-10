"""
=============================================================================
config.py - Central configuration for the OncoLink RAG stack
=============================================================================

All RAG settings can be overridden with environment variables so that the
same code works locally, in CI, and in a deployed Streamlit app.  The values
below are intentionally pragmatic defaults for an educational/demo system;
they are not tuned for clinical production use.

Why a central config module:
    Retrieval behaviour is spread across chunking, embeddings, ChromaDB,
    BM25, reranking, and answer generation.  Keeping the knobs in one place
    makes the pipeline easier to test and explain.

Security note:
    This file never contains API keys.  The NVIDIA key is resolved by
    infrastructure/llm.py from the environment after the UI has merged in
    Streamlit secrets or the sidebar value.
=============================================================================
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Project root = the folder that contains app.py and backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean from the environment, accepting common spellings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    """Read an integer from the environment with a safe fallback."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float from the environment with a safe fallback."""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


@dataclass
class RAGConfig:
    """Runtime settings for chunking, retrieval, and generation."""

    # ---- Chunking -------------------------------------------------------
    # Approximate token budget.  Tokens are counted with tiktoken's cl100k
    # encoding (or a documented fallback), not characters.
    chunk_size: int = field(default_factory=lambda: _env_int("ONCOLINK_CHUNK_SIZE", 700))
    chunk_overlap: int = field(default_factory=lambda: _env_int("ONCOLINK_CHUNK_OVERLAP", 120))

    # ---- Embeddings -----------------------------------------------------
    embedding_model: str = field(
        default_factory=lambda: os.getenv("ONCOLINK_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    )
    embedding_device: str | None = field(
        default_factory=lambda: os.getenv("ONCOLINK_EMBEDDING_DEVICE") or None
    )
    embedding_batch_size: int = field(
        default_factory=lambda: _env_int("ONCOLINK_EMBEDDING_BATCH_SIZE", 32)
    )
    # BGE models recommend this instruction for short-query -> passage search.
    # Set ONCOLINK_QUERY_PREFIX="" to disable it.
    query_prefix: str = field(
        default_factory=lambda: os.getenv(
            "ONCOLINK_QUERY_PREFIX",
            "Represent this sentence for searching relevant passages: ",
        )
    )

    # ---- ChromaDB -------------------------------------------------------
    chroma_dir: str = field(
        default_factory=lambda: os.getenv(
            "ONCOLINK_CHROMA_DIR", str(PROJECT_ROOT / "storage" / "chroma_db")
        )
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("ONCOLINK_CHROMA_COLLECTION", "oncolink_chunks")
    )

    # ---- Retrieval ------------------------------------------------------
    top_k: int = field(default_factory=lambda: _env_int("ONCOLINK_TOP_K", 5))
    candidate_k: int = field(
        default_factory=lambda: _env_int("ONCOLINK_CANDIDATE_K", 20)
    )
    hybrid_search: bool = field(default_factory=lambda: _env_bool("ONCOLINK_HYBRID_SEARCH", True))
    # Cosine similarity threshold in [-1, 1].  None disables thresholding.
    # ChromaDB returns cosine *distance*; the retriever converts it before
    # applying this threshold so we never label a raw distance as a score.
    score_threshold: float | None = field(
        default_factory=lambda: _env_float("ONCOLINK_SCORE_THRESHOLD", -1.0)
    )
    rrf_k: int = field(default_factory=lambda: _env_int("ONCOLINK_RRF_K", 60))

    # ---- Reranking ------------------------------------------------------
    rerank_enabled: bool = field(default_factory=lambda: _env_bool("ONCOLINK_RERANK_ENABLED", False))
    reranker_model: str = field(
        default_factory=lambda: os.getenv("ONCOLINK_RERANKER_MODEL", "BAAI/bge-reranker-base")
    )

    # ---- Generation -----------------------------------------------------
    # Keep this below the model's context window; evidence blocks are small.
    rag_max_context_chunks: int = field(
        default_factory=lambda: _env_int("ONCOLINK_RAG_MAX_CONTEXT_CHUNKS", 8)
    )
    rag_max_tokens: int = field(
        default_factory=lambda: _env_int("ONCOLINK_RAG_MAX_TOKENS", 1800)
    )
    rag_temperature: float = field(
        default_factory=lambda: _env_float("ONCOLINK_RAG_TEMPERATURE", 0.1)
    )
    rag_model: str = field(
        default_factory=lambda: os.getenv(
            "ONCOLINK_RAG_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
        )
    )

    # ---- Logging / safety ----------------------------------------------
    log_raw_questions: bool = field(
        default_factory=lambda: _env_bool("ONCOLINK_LOG_RAW_QUESTIONS", False)
    )

    def effective_score_threshold(self) -> float | None:
        """
        Return the similarity threshold, or None when filtering is disabled.

        The environment default is -1.0 because cosine similarity can range
        from -1 to 1.  A value <= -1 means "keep everything".
        """
        if self.score_threshold is None:
            return None
        if self.score_threshold <= -1.0:
            return None
        return self.score_threshold

    def ensure_storage_dir(self) -> Path:
        """Create the ChromaDB storage directory if it does not exist."""
        path = Path(self.chroma_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


# Default config object.  Callers may build their own RAGConfig for tests.
DEFAULT_CONFIG = RAGConfig()
