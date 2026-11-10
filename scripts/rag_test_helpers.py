"""
Shared helpers for the RAG unit tests.

These fakes keep the tests fast and deterministic: no NVIDIA API key, no
embedding-model download, and no external network access are required.
Set ONCOLINK_TEST_REAL_MODELS=1 to run the optional real-model checks.
"""

import hashlib
import json
import os
import re

import numpy as np


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")


def tokens(text):
    result = []
    for token in _TOKEN_RE.findall((text or "").lower()):
        result.append(token.replace("-", "").replace("_", ""))
    return result


class FakeEmbeddingService:
    """Deterministic bag-of-words embedding used only by tests.

    A larger dimension and salted bigram hashes keep accidental collisions
    from dominating short clinical sentences.  This is still a lexical fake,
    not a semantic model; real semantic behaviour is covered by the optional
    real-model test.
    """

    dimension = 2048

    def embed_documents(self, texts, batch_size=None):
        texts = list(texts or [])
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            word_tokens = tokens(text)
            for token in word_tokens:
                index = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:8], 16) % self.dimension
                vectors[row, index] += 1.0
            # Bigrams add a little context but stay secondary to unigrams.
            for first, second in zip(word_tokens, word_tokens[1:]):
                index = int(
                    hashlib.md5(f"B|{first}_{second}".encode("utf-8")).hexdigest()[:8], 16
                ) % self.dimension
                vectors[row, index] += 0.15
            norm = np.linalg.norm(vectors[row])
            if norm > 0:
                vectors[row] /= norm
        return vectors

    def embed_query(self, query):
        if not query or not query.strip():
            raise ValueError("Cannot embed an empty query.")
        return self.embed_documents([query])[0]

    def is_loaded(self):
        return True


class MockCrossEncoder:
    """Simple lexical-overlap scorer with the CrossEncoder.predict shape."""

    def predict(self, pairs, batch_size=16):
        scores = []
        for query, text in pairs:
            query_tokens = set(tokens(query))
            text_tokens = set(tokens(text))
            overlap = len(query_tokens & text_tokens)
            scores.append(overlap / max(1, len(query_tokens)))
        return np.asarray(scores, dtype=np.float32)


def mock_reranker():
    """Build a Reranker that uses the mock cross-encoder."""
    from backend.reranker import Reranker

    return Reranker(model_name="mock-reranker", enabled=True, model_loader=lambda name: MockCrossEncoder())


def make_mock_llm(
    answer="Grounded answer from the evidence.",
    source_ids=None,
    contradictions=None,
    insufficient=False,
    raw_response=None,
):
    """Return a callable with the same signature as the RAG engine's LLM."""

    def _llm(system_prompt, user_prompt):
        if raw_response is not None:
            return raw_response

        ids = source_ids
        if ids is None:
            ids = re.findall(r"\[(E\d+)\]", user_prompt)[:2]
        sources = [
            {"evidence_id": evidence_id, "chunk_id": "", "quote": "supporting quote"}
            for evidence_id in ids
        ]
        return json.dumps(
            {
                "answer": answer,
                "sources": sources,
                "supporting_evidence": ["supporting fact"],
                "unresolved_contradictions": contradictions or [],
                "insufficient_evidence": insufficient,
            }
        )

    return _llm


class MockFile:
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as handle:
            self._bytes = handle.read()

    def read(self):
        return self._bytes


def synthetic_document(
    patient_id,
    document_id,
    filename,
    text,
    page_number=None,
    document_date="",
    file_type="TXT",
):
    """Build a document dict without going through the file loaders."""
    return {
        "patient_id": patient_id,
        "document_id": document_id,
        "filename": filename,
        "file_type": file_type,
        "text": text,
        "pages": [{"page_number": page_number, "text": text}],
        "content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "upload_timestamp": "2026-01-01T00:00:00+00:00",
        "document_date": document_date,
    }
