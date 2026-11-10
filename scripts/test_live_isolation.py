"""Live patient-isolation and workflow verification.

Uses real embeddings, real ChromaDB and the real NVIDIA model.  Two synthetic
patient cases are indexed into one store to prove that every retrieval, prompt
and citation path is scoped correctly.

Run:

    $env:ONCOLINK_TEST_LIVE = "1"
    python scripts/test_live_isolation.py
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from application.intelligence import CLINICAL, RAG, DocumentIntelligenceService
from application.rag import RAGService
from backend.document_loader import process_uploaded_file
from backend.rag_indexer import build_rag_components
from domain.documents import ClinicalDocument
from infrastructure.config import RAGConfig
from infrastructure.embeddings import EmbeddingService
from infrastructure.vector_store import VectorStore

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
    def __init__(self, path, name=None):
        self.name = name or Path(path).name
        self._bytes = Path(path).read_bytes()

    def read(self):
        return self._bytes


class CountingEmbeddings:
    """Proxy around the real embedder that counts embedding batches."""

    def __init__(self, inner):
        self.inner = inner
        self.batches = 0
        self.texts = 0

    def embed_documents(self, texts, batch_size=None):
        texts = list(texts or [])
        self.batches += 1
        self.texts += len(texts)
        return self.inner.embed_documents(texts, batch_size=batch_size)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def synthetic_document(patient_id, filename, text):
    record = {
        "filename": filename,
        "text": text,
        "file_type": "TXT",
        "success": True,
        "pages": [{"page_number": 1, "text": text}],
    }
    return ClinicalDocument.from_processed_file(record, patient_id)


print("=" * 72)
print("LIVE patient isolation and workflow verification")
print("=" * 72)

if os.getenv("ONCOLINK_TEST_LIVE", "").strip() != "1":
    print("  SKIP  Set ONCOLINK_TEST_LIVE=1 to run live verification.")
    raise SystemExit(0)

api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
if not api_key or api_key == "nvapi-your-key-here":
    print("  SKIP  NVIDIA_API_KEY is not configured. Nothing was executed.")
    raise SystemExit(0)

data_dir = PROJECT_ROOT / "data"
temp_dir = tempfile.mkdtemp(prefix="oncolink_live_isolation_")

config = RAGConfig()
config.chroma_dir = temp_dir
config.chunk_size = 180
config.chunk_overlap = 30
config.top_k = 5
config.candidate_k = 12
config.rerank_enabled = False

case_a = "case_live_isolation_a"
case_b = "case_live_isolation_b"

try:
    print("\n--- Real indexing for two cases ---")
    embedder = CountingEmbeddings(EmbeddingService())
    store = VectorStore(persist_dir=temp_dir, collection_name="oncolink_live_isolation")
    components = build_rag_components(
        config=config,
        embedding_service=embedder,
        vector_store=store,
    )

    # Case A: the lung-cancer sample.
    a_record = process_uploaded_file(MockFile(data_dir / "test_patient.txt"))
    doc_a = ClinicalDocument.from_processed_file(a_record, case_a)

    # Case B: a synthetic record with a unique marker and a different value.
    doc_b = synthetic_document(
        case_b,
        "case_b_oncology.txt",
        "Oncology note. Ovarian carcinoma. "
        "Unique marker xanthochromia. BRCA2 pathogenic variant detected. "
        "PD-L1 expression is 99 percent. CA-125 is 900 U/mL.",
    )

    service_a = RAGService(config=config, components=components)
    service_a.set_documents([doc_a])
    index_a = service_a.ensure_index()

    service_b = RAGService(config=config, components=components)
    service_b.set_documents([doc_b])
    index_b = service_b.ensure_index()

    check("Case A indexes", index_a.get("success"), index_a.get("errors"))
    check("Case B indexes", index_b.get("success"), index_b.get("errors"))
    check("Case A has chunks", store.count(case_a) > 0, store.count(case_a))
    check("Case B has chunks", store.count(case_b) > 0, store.count(case_b))
    check(
        "Case A index lists only its own document",
        [item["filename"] for item in store.list_documents(case_a)] == ["test_patient.txt"],
        store.list_documents(case_a),
    )
    check(
        "Case B index lists only its own document",
        [item["filename"] for item in store.list_documents(case_b)] == ["case_b_oncology.txt"],
        store.list_documents(case_b),
    )

    # ---- No cross-patient retrieval -----------------------------------
    print("\n--- Cross-patient retrieval ---")
    retriever = components["hybrid_retriever"]

    a_query = retriever.retrieve(
        case_a,
        "xanthochromia BRCA2 CA-125",
        top_k=8,
        mode="hybrid",
        candidate_k=12,
    )
    b_query = retriever.retrieve(
        case_b,
        "xanthochromia BRCA2 CA-125",
        top_k=8,
        mode="hybrid",
        candidate_k=12,
    )

    check("Case A retrieval succeeds", a_query.get("success"), a_query.get("error"))
    check("Case B retrieval succeeds", b_query.get("success"), b_query.get("error"))
    check(
        "Case A retrieval never returns Case B chunks",
        all(item.get("patient_id") == case_a for item in a_query.get("results", [])),
        [(item.get("patient_id"), item.get("filename")) for item in a_query.get("results", [])],
    )
    check(
        "Case B retrieval never returns Case A chunks",
        all(item.get("patient_id") == case_b for item in b_query.get("results", [])),
        [(item.get("patient_id"), item.get("filename")) for item in b_query.get("results", [])],
    )
    check(
        "Case B query finds its unique marker",
        any("xanthochromia" in (item.get("text") or "").lower() for item in b_query.get("results", [])),
        [item.get("text", "")[:80] for item in b_query.get("results", [])],
    )

    # ---- No cross-patient answers or citations ------------------------
    print("\n--- Cross-patient answers and citations ---")
    answer_a = service_a.ask(
        patient_id=case_a,
        question="What is the xanthochromia result and the BRCA2 status?",
        top_k=5,
        mode="hybrid",
        rerank=False,
    )
    check("Case A answer completes", answer_a.get("success"), answer_a.get("error"))
    check(
        "Case A answer cites no Case B document",
        all(
            source.get("filename") != "case_b_oncology.txt"
            for source in answer_a.get("sources", [])
        ),
        answer_a.get("sources"),
    )
    check(
        "Case A evidence contains no Case B marker",
        all(
            "xanthochromia" not in (item.get("text") or "").lower()
            for item in answer_a.get("retrieved_evidence", [])
        ),
    )
    print("  A answer:", (answer_a.get("answer") or "").replace("\n", " ")[:220])

    answer_b = service_b.ask(
        patient_id=case_b,
        question="What is the PD-L1 value for this case?",
        top_k=5,
        mode="hybrid",
        rerank=False,
    )
    check("Case B answer succeeds", answer_b.get("success"), answer_b.get("error"))
    check(
        "Case B answer cites only its own document",
        bool(answer_b.get("sources"))
        and all(source.get("filename") == "case_b_oncology.txt" for source in answer_b.get("sources", [])),
        answer_b.get("sources"),
    )
    check(
        "Case B answer contains no Case A document or PD-L1 value",
        "test_patient.txt" not in str(answer_b)
        and "45%" not in (answer_b.get("answer") or ""),
        str(answer_b)[:400],
    )
    print("  B answer:", (answer_b.get("answer") or "").replace("\n", " ")[:220])

    # ---- Incremental indexing and stale chunk replacement -------------
    print("\n--- Incremental indexing ---")
    batches_before_second = embedder.batches
    second_a = service_a.ensure_index()
    check("Repeat index is cached", second_a.get("cached") is True, second_a)
    check("Unchanged Case A document is skipped", second_a.get("skipped") == 1, second_a)
    check(
        "Unchanged Case A document is not re-embedded",
        embedder.batches == batches_before_second,
        (batches_before_second, embedder.batches),
    )

    updated_text = a_record["text"].replace("PD-L1: 45% positive", "PD-L1: 78% positive")
    # Rebuild the per-page text too: chunking uses ``pages`` when it is present,
    # so reusing the old page text would index the previous content again.
    updated_pages = [
        {
            **page,
            "text": (page.get("text") or "").replace(
                "PD-L1: 45% positive", "PD-L1: 78% positive"
            ),
        }
        for page in a_record.get("pages") or []
    ]
    updated_a = ClinicalDocument.from_processed_file(
        {
            "filename": a_record["filename"],
            "text": updated_text,
            "file_type": a_record["file_type"],
            "success": True,
            "pages": updated_pages,
        },
        case_a,
    )
    check("Updated document keeps its stable ID", updated_a.document_id == doc_a.document_id)
    check("Updated document has a new content hash", updated_a.content_hash != doc_a.content_hash)

    service_a.set_documents([updated_a])
    update_result = service_a.ensure_index()
    check("Changed document is re-indexed as an update", update_result.get("updated") == 1, update_result)

    stored = store.get_all_chunks(case_a, [updated_a.document_id])
    stored_text = " ".join(item.get("text", "") for item in stored)
    check("Updated content is retrievable", "78%" in stored_text, stored_text[:300])
    check("Stale content is removed", "45%" not in stored_text, stored_text[:300])
    metadata = store.get_document_metadata(case_a, updated_a.document_id)
    check(
        "Document metadata points at the new hash",
        metadata.get("content_hash") == updated_a.content_hash,
        metadata.get("content_hash"),
    )

    # ---- Case deletion is scoped --------------------------------------
    print("\n--- Scoped deletion ---")
    count_b_before = store.count(case_b)
    deleted = service_a.remove_index(case_a)
    check("Deleting Case A removes chunks", deleted > 0, deleted)
    check("Case A index is empty after deletion", store.count(case_a) == 0, store.count(case_a))
    check("Case B index is untouched", store.count(case_b) == count_b_before, store.count(case_b))

    # ---- Branch failure independence (real authentication failures) ----
    print("\n--- Branch failure independence ---")
    case_c = "case_live_branch_c"
    branch_service = DocumentIntelligenceService(
        patient_id=case_c,
        rag_service=RAGService(config=config, components=components),
    )
    branch_service.ingest([MockFile(data_dir / "test_patient.txt", "branch_case.txt")])

    clinical_ok = branch_service.run_clinical()
    check("Clinical branch succeeds with a valid key", clinical_ok.success, clinical_ok.errors)
    check("Clinical stage reports success", branch_service.stage_status(CLINICAL).successful)

    saved_key = os.environ.get("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = "nvapi-invalid-key-for-failure-test"
    try:
        failed_run = branch_service.run_clinical()
        check("Clinical branch fails with an invalid key", not failed_run.success)
        check("Clinical stage reports failure", branch_service.stage_status(CLINICAL).state.value == "failed")
        check(
            "Clinical failure does not mark RAG failed",
            branch_service.stage_status(RAG).state.value != "failed",
            branch_service.stage_status(RAG).as_dict(),
        )
    finally:
        if saved_key:
            os.environ["NVIDIA_API_KEY"] = saved_key

    rag_after = branch_service.ensure_rag_index()
    check("RAG still indexes after clinical failure", rag_after.get("success"), rag_after.get("errors"))
    check("RAG stage succeeds independently", branch_service.stage_status(RAG).successful)
    check(
        "Clinical stage stays failed after RAG success",
        branch_service.stage_status(CLINICAL).state.value == "failed",
    )

    case_d = "case_live_branch_d"
    # The engine caches its OpenAI client on first use, so the RAG-failure
    # branch needs its own component stack; otherwise the earlier valid client
    # would mask the invalid key.
    rag_failure_components = build_rag_components(
        config=config,
        embedding_service=embedder,
        vector_store=store,
    )
    rag_failure_service = DocumentIntelligenceService(
        patient_id=case_d,
        rag_service=RAGService(config=config, components=rag_failure_components),
    )
    rag_failure_service.ingest([MockFile(data_dir / "progress_note.txt", "branch_rag.txt")])
    clinical_d = rag_failure_service.run_clinical()
    check("Clinical branch succeeds before the RAG failure", clinical_d.success, clinical_d.errors)

    saved_key = os.environ.get("NVIDIA_API_KEY")
    os.environ["NVIDIA_API_KEY"] = "nvapi-invalid-key-for-failure-test"
    try:
        failed_answer = rag_failure_service.ask("What is the PD-L1 value?")
        check("RAG answer fails with an invalid key", not failed_answer.get("success"), failed_answer.get("error"))
        check("RAG stage reports failure", rag_failure_service.stage_status(RAG).state.value == "failed")
        check(
            "RAG failure does not discard clinical data",
            rag_failure_service.clinical_run is not None
            and rag_failure_service.clinical_run.success,
        )
        check(
            "Clinical stage stays successful after RAG failure",
            rag_failure_service.stage_status(CLINICAL).successful,
        )
    finally:
        if saved_key:
            os.environ["NVIDIA_API_KEY"] = saved_key

finally:
    shutil.rmtree(temp_dir, ignore_errors=True)

print()
print("=" * 72)
print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
print("=" * 72)
raise SystemExit(1 if FAIL else 0)
