"""Live reranker verification with the real cross-encoder model.

The real model is only used when it is already cached or when
``ONCOLINK_PROVISION_RERANKER=1`` explicitly allows a download.  Otherwise the
section is reported as SKIPPED, never PASSED.

Run:

    $env:ONCOLINK_TEST_LIVE = "1"
    $env:ONCOLINK_PROVISION_RERANKER = "1"   # optional download
    python scripts/test_live_reranker.py
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from application.rag import RAGService
from backend.document_loader import process_uploaded_file
from domain.documents import ClinicalDocument
from infrastructure.config import RAGConfig
from infrastructure.reranker import Reranker

PASS = 0
FAIL = 0
SKIP = 0


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


def skip(name, reason):
    global SKIP
    SKIP += 1
    print(f"  SKIP  {name} ({reason})")


def model_cached(model_name):
    """Return True when the cross-encoder config is present in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(model_name, "config.json")
        return isinstance(result, str) and Path(result).exists()
    except Exception:
        return False


class MockFile:
    def __init__(self, path):
        self.name = Path(path).name
        self._bytes = Path(path).read_bytes()

    def read(self):
        return self._bytes


print("=" * 72)
print("LIVE reranker verification (real cross-encoder)")
print("=" * 72)

if os.getenv("ONCOLINK_TEST_LIVE", "").strip() != "1":
    print("  SKIP  Set ONCOLINK_TEST_LIVE=1 to run live verification.")
    raise SystemExit(0)

model_name = os.getenv("ONCOLINK_RERANKER_MODEL", "BAAI/bge-reranker-base")
cached = model_cached(model_name)
print(f"  model  : {model_name}")
print(f"  cached : {cached}")

if not cached and os.getenv("ONCOLINK_PROVISION_RERANKER", "").strip() != "1":
    skip("Real reranker checks", "model is not cached; set ONCOLINK_PROVISION_RERANKER=1 to download")
    print()
    print("=" * 72)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 72)
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# Real cross-encoder ordering
# ---------------------------------------------------------------------------
print("\n--- Real cross-encoder scoring ---")

candidates = [
    {
        "chunk_id": "c1",
        "filename": "pathology.txt",
        "page_number": 1,
        "text": "Immunohistochemistry reports PD-L1 expression of 45 percent in the tumour cells.",
    },
    {
        "chunk_id": "c2",
        "filename": "progress.txt",
        "page_number": 2,
        "text": "The progress note records PD-L1 at 60 percent positive after therapy.",
    },
    {
        "chunk_id": "c3",
        "filename": "labs.txt",
        "page_number": None,
        "text": "Haemoglobin was 12.8 g/dL, platelets 245 x 10^3/uL and creatinine 0.9 mg/dL.",
    },
    {
        "chunk_id": "c4",
        "filename": "admin.txt",
        "page_number": None,
        "text": "The clinic waiting room was renovated and the parking garage closes at 8 pm.",
    },
]

start = time.perf_counter()
reranker = Reranker(model_name=model_name, enabled=True)
result = reranker.rerank("What PD-L1 percentage is reported?", candidates, top_k=4)
load_seconds = time.perf_counter() - start

if not result.get("reranked"):
    skip("Real reranker checks", "model could not be loaded in this environment")
    print("  warnings:", result.get("warnings"))
    print()
    print("=" * 72)
    print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
    print("=" * 72)
    raise SystemExit(0)

check("Real reranker loads and runs", result.get("reranked") is True)
check("Real reranker used cross-encoder scores", result.get("score_type") == "cross_encoder_score")
ordered = result["results"]
check("Real reranker returns every candidate", len(ordered) == len(candidates), len(ordered))
check("Real reranker puts a PD-L1 passage first", "PD-L1" in ordered[0]["text"], ordered[0]["text"])
check(
    "Real reranker ranks the irrelevant admin passage last",
    "parking" in ordered[-1]["text"],
    ordered[-1]["text"],
)
scores = [item["rerank_score"] for item in ordered]
check("Real reranker scores are sorted descending", scores == sorted(scores, reverse=True), scores)
check(
    "Relevant passages outscore irrelevant ones",
    min(ordered[0]["rerank_score"], ordered[1]["rerank_score"])
    > max(ordered[2]["rerank_score"], ordered[3]["rerank_score"]),
    scores,
)
print(f"  load+score: {load_seconds:.1f}s")
for item in ordered:
    print(f"    {item['rerank_score']:+.4f}  {item['filename']}: {item['text'][:60]}")

# ---------------------------------------------------------------------------
# Reranked RAG answer keeps patient scoping and citation validation
# ---------------------------------------------------------------------------
print("\n--- Reranked RAG answer (real model) ---")

api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
if not api_key or api_key == "nvapi-your-key-here":
    skip("Reranked RAG answer", "NVIDIA_API_KEY is not configured")
else:
    data_dir = PROJECT_ROOT / "data"
    temp_dir = tempfile.mkdtemp(prefix="oncolink_live_rerank_")
    config = RAGConfig()
    config.chroma_dir = temp_dir
    config.chunk_size = 180
    config.chunk_overlap = 30
    config.top_k = 5
    config.candidate_k = 12

    case_id = "case_live_rerank"
    try:
        record = process_uploaded_file(MockFile(data_dir / "test_patient.txt"))
        document = ClinicalDocument.from_processed_file(record, case_id)

        service = RAGService(config=config)
        service.set_documents([document])
        index_result = service.ensure_index()
        check("Rerank case indexes", index_result.get("success"), index_result.get("errors"))

        start = time.perf_counter()
        answer = service.ask(
            patient_id=case_id,
            question="What PD-L1 percentage and ECOG status are documented?",
            top_k=4,
            mode="hybrid",
            rerank=True,
        )
        seconds = time.perf_counter() - start

        check("Reranked answer succeeds", answer.get("success"), answer.get("error"))
        check("Reranked answer actually used the reranker", answer.get("reranked") is True, answer.get("reranked"))
        check("Reranked answer has validated citations", len(answer.get("sources") or []) > 0)
        check(
            "Reranked evidence stays inside the case",
            all(item.get("patient_id") == case_id for item in answer.get("retrieved_evidence", [])),
        )
        check(
            "Reranked citations point at retrieved chunks",
            all(
                source.get("chunk_id")
                in {item.get("chunk_id") for item in answer.get("retrieved_evidence", [])}
                for source in answer.get("sources", [])
            ),
        )
        print(f"  answer: {seconds:.1f}s, sources={len(answer.get('sources') or [])}")
        print("  " + (answer.get("answer") or "").replace("\n", " ")[:300])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

print()
print("=" * 72)
print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
print("=" * 72)
raise SystemExit(1 if FAIL else 0)
