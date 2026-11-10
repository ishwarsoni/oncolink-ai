"""Tests for backend/vector_store.py.

Uses an in-memory ChromaDB client so no persistent storage is touched.
No embedding model or API key is required.
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.vector_store import VectorStore

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


def make_chunks(patient_id, document_id, filename, texts, page_number=None,
                section_heading="", content_hash="hash-a"):
    chunks = []
    for index, text in enumerate(texts):
        chunks.append({
            "chunk_id": f"{document_id}-{index}",
            "chunk_index": index,
            "text": text,
            "patient_id": patient_id,
            "document_id": document_id,
            "filename": filename,
            "file_type": "TXT",
            "page_number": page_number if page_number is not None else -1,
            "section_heading": section_heading,
            "upload_timestamp": "2026-01-01T00:00:00+00:00",
            "document_date": "2026-01-01",
            "content_hash": content_hash,
            "token_count": len(text.split()),
            "char_count": len(text),
        })
    return chunks


def embeddings(count, seed=0):
    """Deterministic unit vectors with a small variation per row."""
    vectors = np.zeros((count, 8), dtype=np.float32)
    for row in range(count):
        vectors[row, (row + seed) % 8] = 1.0
    return vectors


print("=" * 70)
print("Vector store tests")
print("=" * 70)

try:
    import chromadb

    client = chromadb.EphemeralClient()
except Exception as error:  # pragma: no cover
    print(f"  SKIP  ChromaDB not available: {error}")
    raise SystemExit(0)


def fresh_store(name):
    return VectorStore(client=client, collection_name=name)


# ---- Insert and count --------------------------------------------------
store = fresh_store("test_basic")
check("New collection is empty", store.count() == 0, store.count())

chunks = make_chunks("case_a", "doc1", "report.txt", ["EGFR mutation positive", "PD-L1 45 percent"])
stored = store.upsert_chunks("case_a", chunks, embeddings(len(chunks)))
check("Upsert stores every chunk", stored == 2, stored)
check("Count reflects stored chunks", store.count("case_a") == 2, store.count("case_a"))

# ---- Query and metadata ------------------------------------------------
results = store.query("case_a", embeddings(1, seed=0)[0], top_k=2)
check("Query returns results", len(results) == 2, len(results))
check("Query results include similarity", all("similarity" in r for r in results))
check("Similarity is in [-1, 1]", all(-1.0 <= r["similarity"] <= 1.0 for r in results))
check("Query results include distance", all("distance" in r for r in results))

first = results[0]
check("Metadata patient_id preserved", first["patient_id"] == "case_a", first.get("patient_id"))
check("Metadata document_id preserved", first["document_id"] == "doc1", first.get("document_id"))
check("Metadata filename preserved", first["filename"] == "report.txt", first.get("filename"))
check("Missing page becomes None", first["page_number"] is None, first.get("page_number"))
check("Chunk ID preserved", bool(first["chunk_id"]), first.get("chunk_id"))

# Page number is preserved when supplied.
page_store = fresh_store("test_pages")
page_chunks = make_chunks("case_p", "docp", "scan.pdf", ["Page one content"], page_number=3)
page_store.upsert_chunks("case_p", page_chunks, embeddings(1))
page_result = page_store.query("case_p", embeddings(1)[0], top_k=1)
check("Real page number is preserved", page_result[0]["page_number"] == 3, page_result[0]["page_number"])

# ---- Upsert deduplication ---------------------------------------------
store.upsert_chunks("case_a", chunks, embeddings(len(chunks)))
check("Re-upsert does not duplicate chunks", store.count("case_a") == 2, store.count("case_a"))

# ---- Document existence / content hash --------------------------------
check(
    "document_exists matches content hash",
    store.document_exists("case_a", "doc1", content_hash="hash-a"),
)
check(
    "document_exists rejects a changed hash",
    not store.document_exists("case_a", "doc1", content_hash="hash-b"),
)

# ---- list_documents ----------------------------------------------------
documents = store.list_documents("case_a")
check("list_documents returns one document", len(documents) == 1, documents)
check("list_documents reports chunk count", documents[0]["chunk_count"] == 2, documents[0])

# ---- Document filter ---------------------------------------------------
second_doc = make_chunks("case_a", "doc2", "note.txt", ["ALK negative"], content_hash="hash-c")
store.upsert_chunks("case_a", second_doc, embeddings(1, seed=4))
filtered = store.get_all_chunks("case_a", document_ids=["doc2"])
check("Document filter returns only that document", len(filtered) == 1 and filtered[0]["document_id"] == "doc2", filtered)

# ---- Patient isolation -------------------------------------------------
other = make_chunks("case_b", "doc3", "other.txt", ["HER2 amplified"])
store.upsert_chunks("case_b", other, embeddings(1, seed=2))
check("Second patient stored", store.count("case_b") == 1, store.count("case_b"))

isolated = store.query("case_a", embeddings(1, seed=2)[0], top_k=10)
check(
    "Query never returns another patient's chunks",
    all(r["patient_id"] == "case_a" for r in isolated),
    [r["patient_id"] for r in isolated],
)
all_a = store.get_all_chunks("case_a")
check("get_all_chunks is patient scoped", all(r["patient_id"] == "case_a" for r in all_a))
check("list_documents is patient scoped", {d["document_id"] for d in store.list_documents("case_a")} == {"doc1", "doc2"})

# ---- Signatures --------------------------------------------------------
signature_before = store.patient_signature("case_a")
store.upsert_chunks("case_a", make_chunks("case_a", "doc4", "new.txt", ["KRAS wild type"], content_hash="hash-d"), embeddings(1, seed=5))
signature_after = store.patient_signature("case_a")
check("Patient signature changes after indexing", signature_before != signature_after)

# ---- Deletion ----------------------------------------------------------
check("delete_document returns True", store.delete_document("case_a", "doc4"))
check("delete_document removed the chunks", store.count("case_a") == 3, store.count("case_a"))
check("delete_document is idempotent", not store.delete_document("case_a", "doc4"))

deleted = store.delete_patient("case_a")
check("delete_patient removes all chunks", deleted == 3, deleted)
check("delete_patient leaves other patients intact", store.count("case_b") == 1, store.count("case_b"))

# ---- Validation and empty input ---------------------------------------
check("Empty chunk list returns 0", store.upsert_chunks("case_a", [], np.zeros((0, 8))) == 0)

try:
    store.upsert_chunks("", chunks, embeddings(len(chunks)))
    check("Missing patient_id is rejected", False)
except ValueError:
    check("Missing patient_id is rejected", True)

mismatched = make_chunks("case_other", "docx", "x.txt", ["text"])
try:
    store.upsert_chunks("case_a", mismatched, embeddings(1))
    check("Mismatched chunk patient_id is rejected", False)
except ValueError:
    check("Mismatched chunk patient_id is rejected", True)

check("Count for unknown patient is 0", store.count("case_unknown") == 0)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
