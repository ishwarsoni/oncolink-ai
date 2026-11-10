"""Tests for backend/retriever.py (semantic retrieval).

Uses the deterministic FakeEmbeddingService and an in-memory ChromaDB
client, so no model download or API key is needed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import RAGConfig
from backend.retriever import SemanticRetriever
from backend.vector_store import VectorStore
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
print("Semantic retriever tests")
print("=" * 70)

import chromadb

client = chromadb.EphemeralClient()
store = VectorStore(client=client, collection_name="test_retriever")
embeddings = FakeEmbeddingService()

config = RAGConfig()
config.top_k = 3

# Synthetic clinical chunks for one patient.
texts = [
    "DIAGNOSIS: Non-small cell lung cancer, stage IIIA.",
    "BIOMARKERS: EGFR exon 19 deletion mutation detected. ALK negative. PD-L1 45 percent.",
    "TREATMENT: The patient completed four cycles of carboplatin and pemetrexed.",
    "FOLLOW-UP: Repeat CT imaging in three months.",
    "ALLERGIES: Penicillin causes a rash.",
]
chunks = []
for index, text in enumerate(texts):
    chunks.append({
        "chunk_id": f"chunk-{index}",
        "chunk_index": index,
        "text": text,
        "patient_id": "case_a",
        "document_id": "doc1",
        "filename": "oncology_note.txt",
        "file_type": "TXT",
        "page_number": -1,
        "section_heading": text.split(":")[0],
        "upload_timestamp": "2026-01-01T00:00:00+00:00",
        "document_date": "2026-01-01",
        "content_hash": "hash",
        "token_count": len(text.split()),
        "char_count": len(text),
    })

store.upsert_chunks("case_a", chunks, embeddings.embed_documents([c["text"] for c in chunks]))

retriever = SemanticRetriever(embedding_service=embeddings, vector_store=store, config=config)

# ---- Basic retrieval ---------------------------------------------------
result = retriever.retrieve("case_a", "What is the EGFR mutation status?", top_k=3)
check("Retrieval succeeds", result["success"], result)
check("Score type is cosine similarity", result["score_type"] == "cosine_similarity")
check("Returns results", len(result["results"]) > 0, len(result["results"]))
check("Respects top_k", len(result["results"]) <= 3, len(result["results"]))
check("Results are sorted by similarity", all(
    result["results"][i].get("similarity", -1) >= result["results"][i + 1].get("similarity", -1)
    for i in range(len(result["results"]) - 1)
))

top = result["results"][0]
check("Most relevant chunk mentions EGFR", "EGFR" in top["text"], top["text"])
check("Result has structured fields", all(key in top for key in (
    "chunk_id", "text", "filename", "page_number", "similarity", "document_id", "metadata"
)))
check("Similarity is not a raw distance", top["similarity"] >= -1.0 and top["similarity"] <= 1.0)

# ---- Empty query / missing patient ------------------------------------
empty = retriever.retrieve("case_a", "   ")
check("Empty query is rejected", not empty["success"] and empty["error"])
missing = retriever.retrieve("", "EGFR")
check("Missing patient is rejected", not missing["success"] and missing["error"])

# ---- No results for another patient ----------------------------------
other = retriever.retrieve("case_unknown", "EGFR mutation", top_k=3)
check("Unknown patient returns no results", other["success"] and other["results"] == [])
check("No-result warning is present", bool(other["warnings"]))

# ---- Document filter --------------------------------------------------
filtered = retriever.retrieve("case_a", "EGFR mutation", top_k=3, document_ids=["missing-doc"])
check("Document filter with unknown doc returns nothing", filtered["success"] and filtered["results"] == [])

# ---- Score threshold --------------------------------------------------
strict = retriever.retrieve("case_a", "allergy penicillin", top_k=3, score_threshold=0.99)
check("High threshold filters weak matches", all(r["similarity"] >= 0.99 for r in strict["results"]), strict["results"])

# ---- Patient isolation ------------------------------------------------
other_chunks = [{
    "chunk_id": "b-0",
    "chunk_index": 0,
    "text": "EGFR mutation is positive for this other patient.",
    "patient_id": "case_b",
    "document_id": "docb",
    "filename": "other.txt",
    "file_type": "TXT",
    "page_number": -1,
    "section_heading": "BIOMARKERS",
    "upload_timestamp": "2026-01-01T00:00:00+00:00",
    "document_date": "2026-01-01",
    "content_hash": "hashb",
    "token_count": 8,
    "char_count": 50,
}]
store.upsert_chunks("case_b", other_chunks, embeddings.embed_documents([other_chunks[0]["text"]]))
scoped = retriever.retrieve("case_a", "EGFR mutation", top_k=10)
check("Retriever never crosses patients", all(r["patient_id"] == "case_a" for r in scoped["results"]))

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
