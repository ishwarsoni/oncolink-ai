"""Optional integration test with real models and (optionally) the real API.

This test is skipped by default so the unit suite stays fast and offline.
Run it explicitly:

    # Real embeddings + ChromaDB + hybrid retrieval only:
    ONCOLINK_TEST_REAL_MODELS=1 python scripts/test_rag_integration.py

    # Also call NVIDIA Nemotron (requires NVIDIA_API_KEY):
    $env:NVIDIA_API_KEY = "nvapi-..."
    $env:ONCOLINK_TEST_REAL_MODELS = "1"
    python scripts/test_rag_integration.py

The script exits 0 when the requested checks pass or are skipped cleanly.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import RAGConfig
from backend.document_loader import process_uploaded_file
from backend.embedding_service import get_embedding_service
from backend.hybrid_retriever import HybridRetriever
from backend.rag_engine import RAGEngine
from backend.rag_indexer import build_document_record, index_documents
from backend.retriever import SemanticRetriever
from backend.vector_store import VectorStore
from rag_test_helpers import MockFile

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
print("RAG integration test (real models)")
print("=" * 70)

if os.getenv("ONCOLINK_TEST_REAL_MODELS", "").strip() != "1":
    print("  SKIP  Set ONCOLINK_TEST_REAL_MODELS=1 to run this integration test.")
    raise SystemExit(0)

data_dir = PROJECT_ROOT / "data"
sample_files = [data_dir / "test_patient.txt", data_dir / "progress_note.txt"]
missing = [str(path) for path in sample_files if not path.exists()]
if missing:
    print(f"  SKIP  Sample documents are missing: {missing}")
    raise SystemExit(0)

temp_dir = tempfile.mkdtemp(prefix="oncolink_rag_integration_")
patient_id = "case_integration_test"
config = RAGConfig()
config.chroma_dir = temp_dir
config.chunk_size = 160
config.chunk_overlap = 30
config.top_k = 6
config.candidate_k = 12
config.rerank_enabled = False

try:
    # ---- Real document loading ---------------------------------------
    processed = [process_uploaded_file(MockFile(str(path))) for path in sample_files]
    check("Sample documents load", all(item["success"] for item in processed), processed)

    pdf_path = data_dir / "pathology_report.pdf"
    if pdf_path.exists():
        pdf_result = process_uploaded_file(MockFile(str(pdf_path)))
        check("PDF loads", pdf_result["success"], pdf_result.get("text", "")[:120])
        check(
            "PDF extraction keeps 1-based page numbers",
            bool(pdf_result["pages"]) and all(
                isinstance(page.get("page_number"), int) and page["page_number"] >= 1
                for page in pdf_result["pages"]
            ),
            pdf_result.get("pages"),
        )

    # ---- Real embeddings + persistent ChromaDB -----------------------
    print("  Loading real embedding model (this may take a moment)...")
    embeddings = get_embedding_service(
        model_name=config.embedding_model,
        device=config.embedding_device,
        query_prefix=config.query_prefix,
        batch_size=config.embedding_batch_size,
    )
    probe = embeddings.embed_documents(["EGFR mutation positive"])
    check("Real embedding dimension is 384", probe.shape == (1, 384), probe.shape)

    store = VectorStore(persist_dir=config.chroma_dir, collection_name="oncolink_integration")
    documents = [build_document_record(item, patient_id) for item in processed]
    index_result = index_documents(documents, embeddings, store, config)
    check("Real indexing succeeds", index_result["success"], index_result)
    check("Chunks were stored", index_result["chunks_indexed"] > 0, index_result["chunks_indexed"])

    # Re-indexing unchanged files must skip embedding work.
    second_index = index_documents(documents, embeddings, store, config)
    check("Unchanged documents are not re-embedded", second_index["skipped"] == len(documents), second_index)

    # ---- Real hybrid retrieval ---------------------------------------
    semantic = SemanticRetriever(embeddings, store, config)
    retriever = HybridRetriever(semantic, store, config)
    retrieval = retriever.retrieve(
        patient_id,
        "What are the reported PD-L1 values across the patient's documents?",
        top_k=6,
        mode="hybrid",
    )
    check("Real hybrid retrieval succeeds", retrieval["success"], retrieval)
    retrieved_text = " ".join(item["text"] for item in retrieval["results"])
    check("Retrieval finds PD-L1 45%", "45" in retrieved_text and "PD" in retrieved_text, retrieved_text[:300])
    check("Retrieval finds PD-L1 60%", "60" in retrieved_text and "PD" in retrieved_text, retrieved_text[:300])
    check(
        "Retrieval stays inside the patient",
        all(item["patient_id"] == patient_id for item in retrieval["results"]),
    )

    # ---- Real LLM generation (optional) ------------------------------
    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if not api_key or api_key == "nvapi-your-key-here":
        print("  SKIP  Real LLM generation (NVIDIA_API_KEY is not set).")
    else:
        print("  Calling NVIDIA Nemotron for a grounded answer...")
        engine = RAGEngine(retriever=retriever, reranker=None, config=config)
        answer = engine.answer(
            patient_id,
            "What are the reported PD-L1 values across the patient's documents?",
            top_k=6,
            mode="hybrid",
            rerank=False,
        )
        check("Real answer generation succeeds", answer["success"], answer.get("error"))
        check("Real answer is non-empty", bool(answer["answer"]), answer.get("answer"))
        check("Real answer has validated citations", len(answer["sources"]) > 0, answer["sources"])
        check(
            "Real citations point at retrieved chunks",
            all(
                source["chunk_id"] in {item["chunk_id"] for item in answer["retrieved_evidence"]}
                for source in answer["sources"]
            ),
        )
        print("\n  Answer preview:")
        print("  " + (answer["answer"] or "").replace("\n", "\n  ")[:1200])

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
