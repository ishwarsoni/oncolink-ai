"""
=============================================================================
embedding_service.py - Sentence Transformers embedding engine
=============================================================================

This module turns text into dense vectors that ChromaDB can search.  The
model is loaded lazily and cached for the lifetime of the Python process so
Streamlit reruns and multiple questions do not reload it.

Default model:
    BAAI/bge-small-en-v1.5 (384-dimensional, English)

BGE convention:
    For short-query -> passage retrieval, BGE models recommend an instruction
    prefix on the *query*.  Documents are embedded without that prefix.
    The prefix is configurable in infrastructure/config.py.

Failure behaviour:
    Loading a model can fail (no network, insufficient disk, incompatible
    torch).  We raise EmbeddingServiceError with an actionable message
    instead of silently producing meaningless vectors.  The Streamlit UI
    catches this and keeps the structured-extraction features available.
=============================================================================
"""

import logging
import threading
from typing import Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingServiceError(RuntimeError):
    """Raised when the embedding model cannot be loaded or used."""


# A module-level cache means the model is loaded once per process, not once
# per Streamlit rerun.  The lock protects concurrent session access.
_MODEL_CACHE = {}
_MODEL_CACHE_LOCK = threading.Lock()


class EmbeddingService:
    """Generate normalized document and query embeddings."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: Optional[str] = None,
        query_prefix: str = "Represent this sentence for searching relevant passages: ",
        batch_size: int = 32,
        model_loader=None,
    ):
        self.model_name = model_name
        self.device = device
        self.query_prefix = query_prefix or ""
        self.batch_size = max(1, int(batch_size))
        self._model = None
        self._dimension = None
        self._infer_lock = threading.Lock()
        # Tests can inject a fake loader without downloading a real model.
        self._model_loader = model_loader

    # ------------------------------------------------------------------
    # Model lifecycle
    # ------------------------------------------------------------------
    def _load_model(self):
        """Load the SentenceTransformer model once and return it."""
        if self._model is not None:
            return self._model

        cache_key = (self.model_name, self.device)
        with _MODEL_CACHE_LOCK:
            if cache_key in _MODEL_CACHE:
                self._model = _MODEL_CACHE[cache_key]
                return self._model

            try:
                if self._model_loader is not None:
                    model = self._model_loader(self.model_name)
                else:
                    from sentence_transformers import SentenceTransformer

                    model = SentenceTransformer(self.model_name, device=self.device)
            except Exception as error:
                logger.error("Could not load embedding model %s: %s", self.model_name, error)
                raise EmbeddingServiceError(
                    f"Could not load embedding model '{self.model_name}'. "
                    "Check your internet connection and that sentence-transformers "
                    "and torch are installed. You can choose another model with "
                    "ONCOLINK_EMBEDDING_MODEL."
                ) from error

            _MODEL_CACHE[cache_key] = model
            self._model = model
            logger.info("Loaded embedding model %s", self.model_name)
            return self._model

    @property
    def dimension(self) -> int:
        """Dimensionality of the embedding vectors."""
        if self._dimension is None:
            model = self._load_model()
            try:
                # sentence-transformers >= 5 uses get_embedding_dimension(),
                # older versions use get_sentence_embedding_dimension().
                if hasattr(model, "get_embedding_dimension"):
                    self._dimension = int(model.get_embedding_dimension())
                else:
                    self._dimension = int(model.get_sentence_embedding_dimension())
            except Exception:
                # Fall back to encoding a probe string.
                probe = model.encode(
                    ["dimension probe"],
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                )
                self._dimension = int(np.asarray(probe).shape[1])
        return self._dimension

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def embed_documents(self, texts: Iterable[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Embed a batch of document passages."""
        text_list = [str(text) if text is not None else "" for text in (texts or [])]
        if not text_list:
            return np.zeros((0, self.dimension), dtype=np.float32)

        model = self._load_model()
        batch = max(1, int(batch_size or self.batch_size))

        try:
            with self._infer_lock:
                vectors = model.encode(
                    text_list,
                    batch_size=batch,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
        except Exception as error:
            logger.error("Embedding generation failed: %s", error)
            raise EmbeddingServiceError(f"Embedding generation failed: {error}") from error

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        return vectors

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one user question using the BGE query prefix."""
        if not isinstance(query, str) or not query.strip():
            raise EmbeddingServiceError("Cannot embed an empty query.")
        text = f"{self.query_prefix}{query.strip()}" if self.query_prefix else query.strip()
        vector = self.embed_documents([text])[0]
        return vector

    def is_loaded(self) -> bool:
        """Return True when the model has already been loaded in this process."""
        return self._model is not None


def get_embedding_service(
    model_name: str = "BAAI/bge-small-en-v1.5",
    device: Optional[str] = None,
    query_prefix: str = "Represent this sentence for searching relevant passages: ",
    batch_size: int = 32,
) -> EmbeddingService:
    """
    Return a cached EmbeddingService for the given settings.

    The cache key is built from the model name, device and query prefix so
    that different configurations do not accidentally share a model object.
    """
    key = (model_name, device, query_prefix, int(batch_size))
    with _MODEL_CACHE_LOCK:
        service = _MODEL_CACHE.get(("service",) + key)
        if service is None:
            service = EmbeddingService(
                model_name=model_name,
                device=device,
                query_prefix=query_prefix,
                batch_size=batch_size,
            )
            _MODEL_CACHE[("service",) + key] = service
        return service


def clear_embedding_cache():
    """Drop cached services and models.  Intended for tests and diagnostics."""
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()
