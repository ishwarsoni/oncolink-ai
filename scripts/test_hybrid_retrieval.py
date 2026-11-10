"""Tests for backend/hybrid_retriever.py (BM25 + RRF fusion)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.config import RAGConfig
from backend.hybrid_retriever import HybridRetriever, tokenize
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
print("Hybrid retrieval tests")
print("=" * 70)

import chromadb

client = chromadb.EphemeralClient()
store = VectorStore(client=client, collection_name="test_hybrid")
embeddings = FakeEmbeddingService()

config = RAGConfig()
config.top_k = 3
config.candidate_k = 10
config.rrf_k = 60

texts = [
    "DIAGNOSIS: Non-small cell lung cancer, stage IIIA.",
    "BIOMARKERS: EGFR exon 19 deletion detected. ALK negative. PD-L1 45 percent.",
    "TREATMENT: Completed four cycles of carboplatin and pemetrexed.",
    "FOLLOW-UP: Repeat CT imaging in three months.",
    "ALLERGIES: Penicillin causes a rash.",
    "FAMILY HISTORY: Mother had breast cancer.",
]
chunks = []
for index, text in enumerate(texts):
    chunks.append({
        "chunk_id": f"chunk-{index}",
        "chunk_index": index,
        "text": text,
        "patient_id": "case_a",
        "document_id": "doc1",
        "filename": "note.txt",
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

semantic = SemanticRetriever(embedding_service=embeddings, vector_store=store, config=config)
hybrid = HybridRetriever(semantic_retriever=semantic, vector_store=store, config=config)

# ---- Tokenizer --------------------------------------------------------
check("Tokenizer keeps PD-L1 as a token", "pdl1" in tokenize("PD-L1 expression"))
check("Tokenizer keeps ECOG", "ecog" in tokenize("ECOG performance status"))
check("Tokenizer normalizes case", tokenize("EGFR") == ["egfr"])
check("Empty text yields no tokens", tokenize("") == [])

# ---- BM25 keyword search ----------------------------------------------
keyword = hybrid.keyword_search("case_a", "EGFR mutation", top_k=3)
check("BM25 returns keyword matches", len(keyword) > 0, len(keyword))
check("BM25 top result mentions EGFR", "EGFR" in keyword[0]["text"], keyword[0]["text"] if keyword else "")
check("BM25 results include keyword_score", all("keyword_score" in item for item in keyword))
check("BM25 scores are positive", all(item["keyword_score"] > 0 for item in keyword))
check("BM25 ranks are descending", all(
    keyword[i]["keyword_score"] >= keyword[i + 1]["keyword_score"] for i in range(len(keyword) - 1)
))

no_match = hybrid.keyword_search("case_a", "zzzznotaword", top_k=3)
check("BM25 returns nothing for an unknown term", no_match == [])

# ---- Semantic-only mode ----------------------------------------------
semantic_result = hybrid.retrieve("case_a", "EGFR mutation status", top_k=3, mode="semantic")
check("Semantic mode succeeds", semantic_result["success"], semantic_result)
check("Semantic mode reports cosine score type", semantic_result["score_type"] == "cosine_similarity")
check("Semantic mode respects top_k", len(semantic_result["results"]) <= 3)
check("Semantic results carry similarity", all("similarity" in item for item in semantic_result["results"]))

# ---- Hybrid mode ------------------------------------------------------
hybrid_result = hybrid.retrieve("case_a", "EGFR mutation status", top_k=3, mode="hybrid")
check("Hybrid mode succeeds", hybrid_result["success"], hybrid_result)
check("Hybrid mode reports rrf score type", hybrid_result["score_type"] == "rrf")
check("Hybrid mode respects top_k", len(hybrid_result["results"]) <= 3)
check("Hybrid results include rrf_score", all("rrf_score" in item for item in hybrid_result["results"]))
check("Hybrid results are sorted by RRF", all(
    hybrid_result["results"][i]["rrf_score"] >= hybrid_result["results"][i + 1]["rrf_score"]
    for i in range(len(hybrid_result["results"]) - 1)
))

# A chunk that appears in both rankings should beat one that appears in only
# one ranking.  The EGFR chunk matches both semantically and by keyword.
top_ids = [item["chunk_id"] for item in hybrid_result["results"]]
check("Hybrid surfaces the EGFR chunk", "chunk-1" in top_ids, top_ids)

# ---- RRF fusion math --------------------------------------------------
fused = HybridRetriever._reciprocal_rank_fusion(
    semantic_items=[{"chunk_id": "x"}, {"chunk_id": "y"}],
    keyword_items=[{"chunk_id": "y"}, {"chunk_id": "z"}],
    rrf_k=60,
)
by_id = {item["chunk_id"]: item for item in fused}
check("RRF boosts items in both lists", by_id["y"]["rrf_score"] > by_id["x"]["rrf_score"])
check("RRF records semantic rank", by_id["x"]["semantic_rank"] == 1)
check("RRF records keyword rank", by_id["z"]["keyword_rank"] == 2)

# ---- Cache invalidation ----------------------------------------------
hybrid.retrieve("case_a", "EGFR", top_k=3, mode="hybrid")
cache_before = len(hybrid._bm25_cache)
new_chunk = dict(chunks[0])
new_chunk["chunk_id"] = "chunk-new"
new_chunk["text"] = "NEW: EGFR T790M resistance mutation detected."
store.upsert_chunks("case_a", [new_chunk], embeddings.embed_documents([new_chunk["text"]]))
hybrid_result_2 = hybrid.retrieve("case_a", "T790M resistance", top_k=5, mode="hybrid")
new_ids = [item["chunk_id"] for item in hybrid_result_2["results"]]
check("BM25 cache is invalidated after new chunks", "chunk-new" in new_ids, new_ids)

# ---- Validation and isolation ----------------------------------------
check("Empty query rejected", not hybrid.retrieve("case_a", "  ")["success"])
check("Missing patient rejected", not hybrid.retrieve("", "EGFR")["success"])
unknown = hybrid.retrieve("case_unknown", "EGFR", top_k=3, mode="hybrid")
check("Unknown patient has no results", unknown["success"] and unknown["results"] == [])

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
