"""Live RAG verification with the real Nemotron model and real embeddings.

Execution:

    $env:ONCOLINK_TEST_LIVE = "1"
    python scripts/test_live_rag.py

The answer-generation checks use the real model.  Two checks deliberately feed
the answer validator a hand-written LLM payload to prove that fabricated
citations are rejected; they are labelled as citation-validation checks and
are never presented as model output.
"""

import json
import os
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
from backend.rag_engine import RAGEngine
from domain.documents import ClinicalDocument
from infrastructure.config import RAGConfig

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


class MockFile:
    def __init__(self, path):
        self.name = Path(path).name
        self._bytes = Path(path).read_bytes()

    def read(self):
        return self._bytes


def make_document(path, patient_id):
    record = process_uploaded_file(MockFile(path))
    return ClinicalDocument.from_processed_file(record, patient_id)


def ask(service, question, top_k=6):
    start = time.perf_counter()
    result = service.ask(
        patient_id=service.documents[0].patient_id,
        question=question,
        top_k=top_k,
        mode="hybrid",
        rerank=False,
    )
    return result, time.perf_counter() - start


def show_answer(label, result, seconds):
    answer = (result.get("answer") or "").strip().replace("\n", " ")
    print(f"  {label}: {seconds:.1f}s, sources={len(result.get('sources') or [])}")
    print(f"    answer: {answer[:320]}")


print("=" * 72)
print("LIVE RAG verification (real Nemotron, real embeddings, real ChromaDB)")
print("=" * 72)

if os.getenv("ONCOLINK_TEST_LIVE", "").strip() != "1":
    print("  SKIP  Set ONCOLINK_TEST_LIVE=1 to run live verification.")
    raise SystemExit(0)

api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
if not api_key or api_key == "nvapi-your-key-here":
    print("  SKIP  NVIDIA_API_KEY is not configured. Nothing was executed.")
    raise SystemExit(0)

data_dir = PROJECT_ROOT / "data"
required = [
    data_dir / "test_patient.txt",
    data_dir / "progress_note.txt",
    data_dir / "pathology_report.pdf",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    print(f"  SKIP  Sample documents are missing: {missing}")
    raise SystemExit(0)

temp_dir = tempfile.mkdtemp(prefix="oncolink_live_rag_")
config = RAGConfig()
config.chroma_dir = temp_dir
config.chunk_size = 180
config.chunk_overlap = 30
config.top_k = 6
config.candidate_k = 12
config.rerank_enabled = False

try:
    case_lung = "case_live_rag_lung"
    case_breast = "case_live_rag_breast"

    lung_docs = [
        make_document(data_dir / "test_patient.txt", case_lung),
        make_document(data_dir / "progress_note.txt", case_lung),
    ]

    print("\n--- Real indexing ---")
    service = RAGService(config=config)
    service.set_documents(lung_docs)

    start = time.perf_counter()
    index_result = service.ensure_index()
    index_seconds = time.perf_counter() - start
    check("Real embedding indexing succeeds", index_result.get("success"), index_result.get("errors"))
    check("Chunks were stored", index_result.get("chunks_indexed", 0) > 0, index_result)
    check("Both documents were indexed", index_result.get("indexed") == 2, index_result)
    print(f"  index: {index_seconds:.1f}s, chunks={index_result.get('chunks_indexed')}")

    second = service.ensure_index()
    check("Repeat index uses the cache", second.get("cached") is True, second)
    check("Unchanged documents are skipped", second.get("skipped") == 2, second)

    # ---- Real grounded answers ---------------------------------------
    print("\n--- Real answer generation ---")
    answer, seconds = ask(service, "What is the patient's cancer diagnosis and stage?")
    check("Real diagnosis answer succeeds", answer.get("success"), answer.get("error"))
    check("Real diagnosis answer is non-empty", bool((answer.get("answer") or "").strip()))
    check("Real diagnosis answer has citations", len(answer.get("sources") or []) > 0, answer.get("sources"))
    check(
        "Answer cites retrieved evidence only",
        all(
            source.get("chunk_id") in {item.get("chunk_id") for item in answer.get("retrieved_evidence", [])}
            for source in answer.get("sources", [])
        ),
        answer.get("sources"),
    )
    check(
        "All retrieved evidence belongs to the case",
        all(item.get("patient_id") == case_lung for item in answer.get("retrieved_evidence", [])),
    )
    show_answer("diagnosis", answer, seconds)

    multi, seconds = ask(
        service,
        "What treatments has the patient received, and what PD-L1 values are reported across the documents?",
    )
    multi_filenames = {source.get("filename") for source in multi.get("sources", [])}
    check("Multi-document answer succeeds", multi.get("success"), multi.get("error"))
    check(
        "Multi-document answer cites more than one document",
        len(multi_filenames) >= 2,
        multi_filenames,
    )
    show_answer("multi-document", multi, seconds)

    conflict, seconds = ask(
        service,
        "Do the documents report different PD-L1 results? State every value you find.",
    )
    conflict_text = json.dumps(conflict, default=str)
    mentions_both = "45" in conflict_text and "60" in conflict_text
    explicit = bool(conflict.get("unresolved_contradictions")) or mentions_both
    check("Conflicting PD-L1 values are exposed", conflict.get("success") and explicit, conflict_text[:500])
    show_answer("conflict", conflict, seconds)

    insufficient, seconds = ask(
        service,
        "What is the patient's HIV viral load and CD4 count?",
    )
    check("Insufficient-evidence answer returns successfully", insufficient.get("success"), insufficient.get("error"))
    check(
        "Insufficient-evidence flag is set",
        insufficient.get("insufficient_evidence") is True,
        insufficient.get("insufficient_evidence"),
    )
    insufficient_answer = (insufficient.get("answer") or "").lower()
    check(
        "Insufficient-evidence answer does not invent an HIV result",
        not any(term in insufficient_answer for term in ("copies/ml", "copies/ml", "undetectable", "cells/mm3")),
        insufficient.get("answer"),
    )
    show_answer("insufficient", insufficient, seconds)

    # ---- Citation validation (hand-written payload, real retrieval) ---
    print("\n--- Citation validation (deterministic payload, real retrieval) ---")
    retriever = service.components["hybrid_retriever"]

    fabricated = {
        "answer": "The PD-L1 value is 12%.",
        "sources": [
            {
                "evidence_id": "E999",
                "chunk_id": "fabricated-chunk",
                "filename": "invented.pdf",
                "quote": "PD-L1 is 12%.",
            }
        ],
        "supporting_evidence": [],
        "unresolved_contradictions": [],
        "insufficient_evidence": False,
    }
    fabricated_engine = RAGEngine(
        retriever=retriever,
        reranker=None,
        config=config,
        llm_fn=lambda system, user: json.dumps(fabricated),
    )
    fabricated_result = fabricated_engine.answer(case_lung, "What is the PD-L1 value?", top_k=4, rerank=False)
    check("Fabricated citation is rejected", fabricated_result.get("sources") == [], fabricated_result.get("sources"))
    check(
        "Fabricated citation produces a warning",
        any("discarded" in warning.lower() or "did not contain a valid citation" in warning.lower()
            for warning in fabricated_result.get("warnings", [])),
        fabricated_result.get("warnings"),
    )

    valid = {
        "answer": "The PD-L1 value is documented in the cited report.",
        "sources": [
            {
                "evidence_id": "E1",
                "chunk_id": "",
                "filename": "invented.pdf",
                "page_number": 99,
                "quote": "PD-L1",
            }
        ],
        "supporting_evidence": [],
        "unresolved_contradictions": [],
        "insufficient_evidence": False,
    }
    valid_engine = RAGEngine(
        retriever=retriever,
        reranker=None,
        config=config,
        llm_fn=lambda system, user: json.dumps(valid),
    )
    valid_result = valid_engine.answer(case_lung, "What is the PD-L1 value?", top_k=4, rerank=False)
    check("Valid evidence ID produces a citation", len(valid_result.get("sources") or []) == 1, valid_result.get("sources"))
    if valid_result.get("sources"):
        citation = valid_result["sources"][0]
        evidence = {item["evidence_id"]: item for item in valid_result.get("retrieved_evidence", [])}
        check(
            "Citation metadata is rebuilt from stored evidence",
            citation.get("filename") == evidence["E1"].get("filename")
            and citation.get("page_number") == evidence["E1"].get("page_number"),
            citation,
        )

    # ---- PDF page references ------------------------------------------
    print("\n--- Page citations from a real PDF ---")
    pdf_doc = make_document(data_dir / "pathology_report.pdf", case_breast)
    pdf_service = RAGService(config=config)
    pdf_service.set_documents([pdf_doc])
    pdf_index = pdf_service.ensure_index()
    check("PDF case indexes", pdf_index.get("success"), pdf_index.get("errors"))

    pdf_answer, seconds = ask(
        pdf_service,
        "What is the HER2 result and the Nottingham histologic grade?",
        top_k=4,
    )
    pdf_sources = pdf_answer.get("sources") or []
    pdf_pages = [source.get("page_number") for source in pdf_sources]
    check("PDF answer succeeds", pdf_answer.get("success"), pdf_answer.get("error"))
    check("PDF answer cites the pathology report", any(
        source.get("filename") == "pathology_report.pdf" for source in pdf_sources
    ), pdf_sources)
    check(
        "PDF citations use 1-based page numbers",
        bool(pdf_pages) and all(isinstance(page, int) and page >= 1 for page in pdf_pages),
        pdf_pages,
    )
    check(
        "PDF citations include a page label",
        all(bool(source.get("page_label")) for source in pdf_sources),
        [source.get("page_label") for source in pdf_sources],
    )
    show_answer("pdf", pdf_answer, seconds)

finally:
    import shutil

    shutil.rmtree(temp_dir, ignore_errors=True)

print()
print("=" * 72)
print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
print("=" * 72)
raise SystemExit(1 if FAIL else 0)
