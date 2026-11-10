"""Patient isolation, indexing lifecycle, and safety tests.

Verifies that one patient's documents can never be retrieved, cited, or
deleted while acting on another patient.
"""

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import RAGConfig
from backend.hybrid_retriever import HybridRetriever
from backend.rag_engine import RAGEngine
from backend.rag_indexer import (
    build_document_record,
    generate_patient_id,
    index_documents,
    remove_patient_index,
    safe_filename,
)
from backend.retriever import SemanticRetriever
from backend.vector_store import VectorStore
from rag_test_helpers import FakeEmbeddingService, synthetic_document

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
print("Patient isolation tests")
print("=" * 70)

import chromadb

client = chromadb.EphemeralClient()
store = VectorStore(client=client, collection_name="test_isolation")
embeddings = FakeEmbeddingService()

config = RAGConfig()
config.top_k = 5
config.candidate_k = 10
config.chunk_size = 200
config.chunk_overlap = 20
config.rerank_enabled = False

patient_a = "case_aaaaaaaaaaaa"
patient_b = "case_bbbbbbbbbbbb"

doc_a = synthetic_document(
    patient_id=patient_a,
    document_id="doc_a",
    filename="shared_name.txt",
    text="BIOMARKERS:\nPatient A has EGFR exon 19 deletion. PD-L1 45%.\n",
    document_date="2024-01-01",
)
doc_b = synthetic_document(
    patient_id=patient_b,
    document_id="doc_b",
    filename="shared_name.txt",
    text="BIOMARKERS:\nPatient B has ALK rearrangement. PD-L1 90%.\n",
    document_date="2024-02-02",
)

result_a = index_documents([doc_a], embeddings, store, config)
result_b = index_documents([doc_b], embeddings, store, config)
check("Patient A indexed", result_a["success"] and result_a["indexed"] == 1, result_a)
check("Patient B indexed", result_b["success"] and result_b["indexed"] == 1, result_b)
check("Chunks are stored separately", store.count(patient_a) > 0 and store.count(patient_b) > 0)

# ---- Retrieval isolation ----------------------------------------------
semantic = SemanticRetriever(embeddings, store, config)
retriever = HybridRetriever(semantic, store, config)

results_a = retriever.retrieve(patient_a, "ALK rearrangement PD-L1 90", top_k=10, mode="hybrid")
results_b = retriever.retrieve(patient_b, "EGFR exon 19 deletion PD-L1 45", top_k=10, mode="hybrid")

check("Patient A retrieval is non-empty", len(results_a["results"]) > 0)
check("Patient B retrieval is non-empty", len(results_b["results"]) > 0)
check("Patient A never sees Patient B chunks", all(r["patient_id"] == patient_a for r in results_a["results"]))
check("Patient B never sees Patient A chunks", all(r["patient_id"] == patient_b for r in results_b["results"]))
check("Patient A results mention A's biomarker", any("EGFR" in r["text"] for r in results_a["results"]))
check("Patient B results mention B's biomarker", any("ALK" in r["text"] for r in results_b["results"]))

check("get_all_chunks is scoped to A", all(r["patient_id"] == patient_a for r in store.get_all_chunks(patient_a)))
check("list_documents is scoped to A", all(d["filename"] == "shared_name.txt" for d in store.list_documents(patient_a)))
check("Both patients can reuse the same filename", len(store.list_documents(patient_a)) == 1 and len(store.list_documents(patient_b)) == 1)

# ---- RAG prompt isolation ---------------------------------------------
captured = {}


def capture_llm(system_prompt, user_prompt):
    captured["prompt"] = user_prompt
    ids = re.findall(r"\[(E\d+)\]", user_prompt)
    return json.dumps({
        "answer": "Answer based on the retrieved evidence.",
        "sources": [{"evidence_id": ids[0], "chunk_id": "", "quote": "evidence"}] if ids else [],
        "supporting_evidence": [],
        "unresolved_contradictions": [],
        "insufficient_evidence": False,
    })


engine = RAGEngine(retriever=retriever, reranker=None, config=config, llm_fn=capture_llm)
answer_a = engine.answer(patient_a, "What is the biomarker status?", top_k=5, rerank=False)

check("RAG answer for A succeeds", answer_a["success"], answer_a)
check("RAG prompt for A contains A's evidence", "EGFR" in captured["prompt"], captured["prompt"][:200])
check("RAG prompt for A never contains B's data", "ALK rearrangement" not in captured["prompt"])
check("RAG prompt for A never contains B's patient id", patient_b not in captured["prompt"])
check("RAG citations belong to A", all(s["document_id"] == "doc_a" for s in answer_a["sources"]), answer_a["sources"])

# ---- Deletion isolation -----------------------------------------------
deleted = remove_patient_index(store, patient_a)
check("Removing A deletes only A's chunks", deleted > 0, deleted)
check("A has no chunks after removal", store.count(patient_a) == 0)
check("B is untouched after removing A", store.count(patient_b) > 0, store.count(patient_b))
check("B can still be retrieved after removing A", len(retriever.retrieve(patient_b, "ALK", top_k=5)["results"]) > 0)

# ---- Indexing lifecycle -----------------------------------------------
patient_c = "case_cccccccccccc"
doc_c = synthetic_document(
    patient_id=patient_c,
    document_id="doc_c",
    filename="note.txt",
    text="DIAGNOSIS:\nNSCLC stage IIIA.\n",
    document_date="2024-01-01",
)
first = index_documents([doc_c], embeddings, store, config)
second = index_documents([doc_c], embeddings, store, config)
check("First index stores chunks", first["indexed"] == 1 and first["chunks_indexed"] > 0, first)
check("Unchanged document is skipped", second["skipped"] == 1 and second["chunks_indexed"] == 0, second)
check("Unchanged document does not duplicate chunks", store.count(patient_c) == first["chunks_indexed"])

# A changed document replaces the old version.
changed_doc = dict(doc_c)
changed_doc["text"] = "DIAGNOSIS:\nNSCLC stage IV with new liver metastasis.\n"
changed_doc["pages"] = [{"page_number": None, "text": changed_doc["text"]}]
import hashlib

changed_doc["content_hash"] = hashlib.sha256(changed_doc["text"].encode("utf-8")).hexdigest()
third = index_documents([changed_doc], embeddings, store, config)
check("Changed document is updated", third["updated"] == 1, third)
all_chunks_c = store.get_all_chunks(patient_c)
check("Old content is removed on update", all("stage IIIA" not in c["text"] for c in all_chunks_c))
check("New content is stored", any("stage IV" in c["text"] for c in all_chunks_c))

# Force re-index works even when unchanged.
forced = index_documents([changed_doc], embeddings, store, config, force=True)
check("Force re-index does not skip", forced["skipped"] == 0 and forced["updated"] == 1, forced)

# ---- Empty document handling ------------------------------------------
empty_doc = synthetic_document(
    patient_id=patient_c,
    document_id="doc_empty",
    filename="empty.txt",
    text="",
)
empty_result = index_documents([empty_doc], embeddings, store, config)
check("Empty document is skipped safely", empty_result["success"] and empty_result["skipped"] == 1, empty_result)
check("Empty document emits a warning", bool(empty_result["warnings"]))

# ---- Identifier and filename safety -----------------------------------
generated = generate_patient_id()
check("Generated patient id has the expected format", bool(re.fullmatch(r"case_[0-9a-f]{12}", generated)), generated)
check("Generated patient ids are unique", generate_patient_id() != generate_patient_id())

check("safe_filename strips directories", safe_filename("../../etc/passwd") == "passwd", safe_filename("../../etc/passwd"))
check("safe_filename strips Windows paths", safe_filename(r"C:\Users\x\report.pdf") == "report.pdf")
check("safe_filename removes control characters", safe_filename("bad\x00name.txt") == "badname.txt")
check("safe_filename handles empty input", safe_filename("") == "unknown")

# ---- Document record safety -------------------------------------------
record_a = build_document_record({"filename": "report.pdf", "text": "text", "file_type": "PDF"}, patient_a)
record_b = build_document_record({"filename": "report.pdf", "text": "text", "file_type": "PDF"}, patient_b)
check("Same filename yields different document ids per patient", record_a["document_id"] != record_b["document_id"])
check("Document record stores the patient id", record_a["patient_id"] == patient_a)

try:
    build_document_record({"filename": "x.txt", "text": "x"}, "")
    check("Document record requires a patient id", False)
except ValueError:
    check("Document record requires a patient id", True)

# ---- No cross-patient citation leakage --------------------------------
answer_b = engine.answer(patient_b, "What is the biomarker status?", top_k=5, rerank=False)
check("RAG answer for B succeeds", answer_b["success"], answer_b)
check("RAG prompt for B contains B's evidence", "ALK rearrangement" in captured["prompt"])
check("RAG prompt for B never contains A's data", "EGFR exon 19 deletion" not in captured["prompt"])
check("RAG citations belong to B", all(s["document_id"] == "doc_b" for s in answer_b["sources"]), answer_b["sources"])

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
