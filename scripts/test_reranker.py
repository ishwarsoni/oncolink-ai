"""Tests for backend/reranker.py (optional cross-encoder reranking)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.reranker import Reranker
from rag_test_helpers import MockCrossEncoder

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
print("Reranker tests")
print("=" * 70)

candidates = [
    {"chunk_id": "a", "text": "ALLERGIES: Penicillin causes a rash.", "similarity": 0.9},
    {"chunk_id": "b", "text": "BIOMARKERS: EGFR exon 19 deletion detected.", "similarity": 0.5},
    {"chunk_id": "c", "text": "TREATMENT: Carboplatin and pemetrexed.", "similarity": 0.4},
]
query = "What is the EGFR mutation status?"

# ---- Disabled reranker -------------------------------------------------
disabled = Reranker(model_name="unused", enabled=False)
disabled_result = disabled.rerank(query, candidates, top_k=2)
check("Disabled reranker succeeds", disabled_result["success"])
check("Disabled reranker is not reranked", disabled_result["reranked"] is False)
check("Disabled reranker preserves order", [r["chunk_id"] for r in disabled_result["results"]] == ["a", "b"])
check("Disabled reranker applies top_k", len(disabled_result["results"]) == 2)
check("Disabled reranker is not available", disabled.is_available() is False)

# ---- Mock cross-encoder ------------------------------------------------
reranker = Reranker(
    model_name="mock-reranker",
    enabled=True,
    model_loader=lambda name: MockCrossEncoder(),
)
check("Enabled reranker reports availability", reranker.is_available() is True)

result = reranker.rerank(query, candidates, top_k=2)
check("Reranker succeeds", result["success"], result)
check("Reranker flags reranked", result["reranked"] is True)
check("Reranker reports score type", result["score_type"] == "cross_encoder_score")
check("Reranker applies top_k", len(result["results"]) == 2, len(result["results"]))
check("Reranker puts the EGFR chunk first", result["results"][0]["chunk_id"] == "b", result["results"][0])
check("Reranked results carry rerank_score", all("rerank_score" in r for r in result["results"]))
check("Reranked scores are descending", all(
    result["results"][i]["rerank_score"] >= result["results"][i + 1]["rerank_score"]
    for i in range(len(result["results"]) - 1)
))
check("Reranker preserves original metadata", result["results"][0]["similarity"] == 0.5)
check("Reranker overwrites score with rerank score", result["results"][0]["score"] == result["results"][0]["rerank_score"])

# ---- Empty candidates --------------------------------------------------
empty = reranker.rerank(query, [], top_k=2)
check("Empty candidates are handled", empty["success"] and empty["results"] == [] and empty["reranked"] is False)

# ---- Model load failure ------------------------------------------------
def broken_loader(name):
    raise RuntimeError("simulated reranker download failure")


broken = Reranker(model_name="broken-reranker", enabled=True, model_loader=broken_loader)
broken_result = broken.rerank(query, candidates, top_k=2)
check("Load failure does not raise", broken_result["success"] is True)
check("Load failure falls back to original order", [r["chunk_id"] for r in broken_result["results"]] == ["a", "b"])
check("Load failure adds a warning", bool(broken_result["warnings"]))
check("Load failure is not marked reranked", broken_result["reranked"] is False)
check("Broken reranker reports unavailable", broken.is_available() is False)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
