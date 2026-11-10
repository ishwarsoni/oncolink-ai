"""Tests for backend/embedding_service.py."""

import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.embedding_service import EmbeddingService, EmbeddingServiceError
from rag_test_helpers import FakeEmbeddingService

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}")
        if detail:
            print(f"        {detail}")


print("=" * 70)
print("Embedding service tests")
print("=" * 70)

# ---- Deterministic fake embedding -------------------------------------
fake = FakeEmbeddingService()
vectors = fake.embed_documents(
    [
        "EGFR exon 19 deletion detected",
        "PD-L1 expression is 45 percent",
        "No evidence of distant metastasis",
    ]
)
check("Embedding matrix has expected shape", vectors.shape == (3, fake.dimension), vectors.shape)
check("Embeddings are finite", np.isfinite(vectors).all())
check("Embeddings are normalized", np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5))
check("Query embedding has one dimension", fake.embed_query("EGFR").shape == (fake.dimension,))
check("Empty batch returns an empty matrix", fake.embed_documents([]).shape == (0, fake.dimension))
try:
    fake.embed_query("")
    check("Empty query is rejected", False)
except Exception:
    check("Empty query is rejected", True)

# ---- Real EmbeddingService with an injected fake model ----------------
class FakeModel:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def get_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        self.calls += 1
        self.inputs.append(list(texts))
        return np.ones((len(texts), 4), dtype=np.float32)


loader_calls = {"count": 0}
fake_model = FakeModel()


def loader(model_name):
    loader_calls["count"] += 1
    return fake_model


service = EmbeddingService(
    model_name="fake-embedding-model-for-tests",
    query_prefix="QUERY PREFIX: ",
    model_loader=loader,
)
check("Model is loaded lazily", loader_calls["count"] == 0)
check("Dimension is reported", service.dimension == 4)
check("Model is loaded once for dimension", loader_calls["count"] == 1)
service.embed_documents(["first document", "second document"])
service.embed_documents(["third document"])
check("Model is not reloaded for later batches", loader_calls["count"] == 1)
check("Batch encoding was called", fake_model.calls == 2)
service.embed_query("EGFR mutation")
check(
    "Query prefix is applied to queries",
    fake_model.inputs[-1][0].startswith("QUERY PREFIX:"),
    fake_model.inputs[-1][0],
)
check("Document encoding does not apply query prefix", fake_model.inputs[0][0] == "first document")

# ---- Model loading failure is explicit --------------------------------
def broken_loader(model_name):
    raise RuntimeError("simulated model download failure")


broken = EmbeddingService(model_name="broken-model-for-tests", model_loader=broken_loader)
try:
    broken.embed_documents(["text"])
    check("Loading failure raises EmbeddingServiceError", False)
except EmbeddingServiceError:
    check("Loading failure raises EmbeddingServiceError", True)
except Exception as error:
    check("Loading failure raises EmbeddingServiceError", False, repr(error))

# ---- Optional real model test -----------------------------------------
if os.getenv("ONCOLINK_TEST_REAL_MODELS", "").strip() == "1":
    print("\n  Running optional real-model check (BAAI/bge-small-en-v1.5)...")
    from backend.embedding_service import get_embedding_service

    real = get_embedding_service("BAAI/bge-small-en-v1.5")
    related = real.embed_documents(["EGFR exon 19 deletion detected"])
    other = real.embed_documents(["The weather is sunny today"])
    query = real.embed_query("EGFR mutation")
    check("Real model dimension is 384", related.shape == (1, 384), related.shape)
    related_score = float((related[0] * query).sum())
    other_score = float((other[0] * query).sum())
    check(
        "Real semantic similarity is higher for related text",
        related_score > other_score,
        f"related={related_score:.3f} other={other_score:.3f}",
    )
else:
    print("  SKIP  Real-model check (set ONCOLINK_TEST_REAL_MODELS=1 to enable)")

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
