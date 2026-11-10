"""Offline workflow tests for the unified document-intelligence pipeline.

These checks exercise the application layer with deterministic fakes.  They
verify the properties the refactor was meant to guarantee:

    * one read per upload, shared by both branches
    * shared canonical document / patient IDs and page references
    * content-hash indexing (unchanged files are not re-embedded)
    * stale-chunk replacement when a document changes
    * patient isolation and citation ownership
    * independent clinical / RAG failure statuses
    * lazy model loading

No NVIDIA API key, model download or network access is required.
"""

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb

import application.intelligence as intelligence_module
from application.clinical import ClinicalRun
from application.intelligence import (
    CLINICAL,
    INGESTION,
    RAG,
    DocumentIntelligenceService,
)
from application.ingestion import ingest_uploads
from application.rag import RAGService
from backend.chunker import chunk_document
from backend.rag_engine import RAGEngine
from backend.rag_indexer import build_rag_components
from domain.documents import (
    ClinicalDocument,
    build_document_id,
    document_content_hash,
)
from infrastructure.config import RAGConfig
from infrastructure.embeddings import EmbeddingService
from infrastructure.reranker import Reranker
from infrastructure.vector_store import VectorStore
from rag_test_helpers import FakeEmbeddingService, make_mock_llm

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


class FakeUpload:
    """Stand-in for a Streamlit UploadedFile that counts reads."""

    def __init__(self, name, text):
        self.name = name
        self.text = text
        self.read_count = 0

    def read(self):
        self.read_count += 1
        return self.text.encode("utf-8")


class CountingEmbeddingService(FakeEmbeddingService):
    """Fake embedder that records how often documents are embedded."""

    def __init__(self):
        self.calls = 0
        self.texts = 0

    def embed_documents(self, texts, batch_size=None):
        texts = list(texts or [])
        self.calls += 1
        self.texts += len(texts)
        return super().embed_documents(texts, batch_size=batch_size)


def make_document(patient_id, filename, text, pages=None):
    """Build a canonical document through the real projection boundary."""
    record = {
        "filename": filename,
        "text": text,
        "file_type": "TXT",
        "success": True,
        "pages": pages or [{"page_number": 1, "text": text}],
    }
    return ClinicalDocument.from_processed_file(record, patient_id)


def make_components(config, embedder, collection_name):
    """Build a complete RAG stack with fake embeddings and ephemeral storage."""
    store = VectorStore(
        client=chromadb.EphemeralClient(),
        collection_name=collection_name,
    )
    components = build_rag_components(
        config=config,
        embedding_service=embedder,
        vector_store=store,
    )
    components["rag_engine"] = RAGEngine(
        retriever=components["hybrid_retriever"],
        reranker=components["reranker"],
        config=config,
        llm_fn=make_mock_llm(),
    )
    return components


print("=" * 70)
print("Unified document-intelligence workflow tests")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Ingestion reads each upload once
# ---------------------------------------------------------------------------
print("\n--- Ingestion ---")

patient = "case_workflow_a"
uploads = [
    FakeUpload("pathology.txt", "Pathology report. PD-L1 45 percent. EGFR wild type."),
    FakeUpload("labs.txt", "Lab report. Haemoglobin 11.2 g/dL."),
    FakeUpload("empty.txt", "   "),
]
ingested = ingest_uploads(uploads, patient)

check("All uploads produce a display record", len(ingested.display_records) == 3)
check("Each upload is read exactly once", all(item.read_count == 1 for item in uploads))
check("Readable uploads become canonical documents", len(ingested.documents) == 2, ingested.errors)
check("Empty upload is excluded from canonical documents", len(ingested.documents) == 2)
check("Empty upload produces a warning", any("no extractable text" in w for w in ingested.warnings))
check("Canonical documents share the patient ID", all(d.patient_id == patient for d in ingested.documents))
check(
    "Document ID is derived from patient and filename",
    ingested.documents[0].document_id == build_document_id(patient, "pathology.txt"),
)
check(
    "Content hash matches canonical text",
    ingested.documents[0].content_hash == document_content_hash(ingested.documents[0].text),
)

# ---------------------------------------------------------------------------
# 2. Both branches consume the same canonical documents
# ---------------------------------------------------------------------------
print("\n--- Shared document layer ---")

config = RAGConfig()
config.chunk_size = 120
config.chunk_overlap = 20
config.embedding_batch_size = 8
config.rerank_enabled = False

embedder = CountingEmbeddingService()
components = make_components(config, embedder, "workflow_shared")
service = DocumentIntelligenceService(
    patient_id=patient,
    rag_service=RAGService(config=config, components=components),
)
service.ingest(uploads)

check("Ingestion stage succeeded", service.stage_status(INGESTION).successful)
check("RAG service received the canonical documents", service.rag.documents == service.documents)
check(
    "Both branches see the same document IDs",
    [d.document_id for d in service.rag.documents]
    == [d.document_id for d in service.documents],
)

captured_clinical = {}


def fake_extract(documents, client=None):
    captured_clinical["documents"] = list(documents)
    return ClinicalRun(
        combined={
            "success": True,
            "data": {"patient_name": "Test Patient", "diagnosis": "Breast cancer"},
            "errors": [],
            "warnings": [],
        },
        per_document=(),
    )


real_extract = intelligence_module.extract_clinical
intelligence_module.extract_clinical = fake_extract
try:
    clinical_run = service.run_clinical()
finally:
    intelligence_module.extract_clinical = real_extract

check("Clinical stage used the shared documents", captured_clinical.get("documents") == list(service.documents))
check("Clinical stage succeeded", clinical_run.success and service.stage_status(CLINICAL).successful)

captured_index = {}


def fake_index(force=False):
    captured_index["ids"] = [d.document_id for d in service.rag.documents]
    return {
        "success": True,
        "indexed": 1,
        "updated": 0,
        "skipped": 0,
        "chunks_indexed": 1,
        "errors": [],
        "warnings": [],
        "documents": [],
    }


service.rag.ensure_index = fake_index
try:
    service.ensure_rag_index()
finally:
    service.rag.ensure_index = RAGService.ensure_index.__get__(service.rag, RAGService)

check(
    "RAG stage used the same document IDs",
    captured_index.get("ids") == [d.document_id for d in service.documents],
)
check("RAG stage succeeded", service.stage_status(RAG).successful)

# ---------------------------------------------------------------------------
# 3. Page references survive chunking
# ---------------------------------------------------------------------------
print("\n--- Page references ---")

paged = make_document(
    patient,
    "paged.txt",
    "Page one. Tumour size 22 mm.\nPage two. No lymphovascular invasion.",
    pages=[
        {"page_number": 1, "text": "Tumour size 22 mm in the right breast."},
        {"page_number": 2, "text": "No lymphovascular invasion is identified."},
    ],
)
chunks = chunk_document(paged.as_index_record(), chunk_size=20, chunk_overlap=5)
page_numbers = {chunk.get("page_number") for chunk in chunks}
check("Chunking keeps 1-based page numbers", page_numbers == {1, 2}, page_numbers)
check("Both pages produced evidence", len(chunks) >= 2)
check("Page numbers are never zero-based", 0 not in page_numbers)

# ---------------------------------------------------------------------------
# 4. Content-hash indexing and stale-chunk replacement
# ---------------------------------------------------------------------------
print("\n--- Incremental indexing ---")

index_config = RAGConfig()
index_config.chunk_size = 80
index_config.chunk_overlap = 10
index_config.embedding_batch_size = 4
index_config.rerank_enabled = False

index_embedder = CountingEmbeddingService()
index_components = make_components(index_config, index_embedder, "workflow_incremental")
index_service = RAGService(config=index_config, components=index_components)

first = make_document(patient, "report.txt", "Initial report. PD-L1 45 percent.")
index_service.set_documents([first])
first_result = index_service.ensure_index()

check("First index succeeds", first_result.get("success"), first_result)
check("First index embeds the document", first_result.get("indexed") == 1)
calls_after_first = index_embedder.calls
check("First index called the embedder once", calls_after_first == 1)

second_result = index_service.ensure_index()
check("Repeated index succeeds", second_result.get("success"), second_result)
check("Unchanged document is skipped", second_result.get("skipped") == 1, second_result)
check("Unchanged document is not re-embedded", index_embedder.calls == calls_after_first)

updated = make_document(patient, "report.txt", "Updated report. PD-L1 60 percent.")
index_service.set_documents([updated])
updated_result = index_service.ensure_index()

check("Updated document is re-indexed", updated_result.get("updated") == 1, updated_result)
check("Update re-embeds the changed text", index_embedder.calls > calls_after_first)

stored_chunks = index_components["vector_store"].get_all_chunks(
    patient, [updated.document_id]
)
stored_text = " ".join(chunk.get("text", "") for chunk in stored_chunks)
check("New content is stored", "60 percent" in stored_text)
check("Stale content is removed", "45 percent" not in stored_text)
check(
    "Document metadata points at the new hash",
    index_components["vector_store"]
    .get_document_metadata(patient, updated.document_id)
    .get("content_hash")
    == updated.content_hash,
)

# ---------------------------------------------------------------------------
# 5. Patient isolation and citation ownership
# ---------------------------------------------------------------------------
print("\n--- Patient isolation ---")

patient_a = "case_isolation_a"
patient_b = "case_isolation_b"
doc_a = make_document(patient_a, "a.txt", "Patient A report. PD-L1 45 percent.")
doc_b = make_document(patient_b, "b.txt", "Patient B report. PD-L1 99 percent.")

iso_config = RAGConfig()
iso_config.chunk_size = 80
iso_config.chunk_overlap = 10
iso_config.embedding_batch_size = 4
iso_config.rerank_enabled = False
iso_components = make_components(iso_config, FakeEmbeddingService(), "workflow_isolation")

service_a = RAGService(config=iso_config, components=iso_components)
service_a.set_documents([doc_a])
service_a.ensure_index()

service_b = RAGService(config=iso_config, components=iso_components)
service_b.set_documents([doc_b])
service_b.ensure_index()

answer_a = service_a.ask(
    patient_id=patient_a,
    question="What is the PD-L1 value?",
    top_k=5,
    mode="hybrid",
    rerank=False,
)

check("Answer for patient A succeeds", answer_a.get("success"), answer_a)
check("Answer has validated citations", bool(answer_a.get("sources")), answer_a)
check(
    "Every citation belongs to patient A's document",
    all(source.get("filename") == "a.txt" for source in answer_a.get("sources", [])),
    answer_a.get("sources"),
)
check(
    "Patient B evidence never appears",
    all(
        item.get("filename") != "b.txt"
        for item in answer_a.get("retrieved_evidence", [])
    ),
)
check(
    "Every evidence chunk carries patient A's scope",
    iso_components["vector_store"].count(patient_b) > 0,
)

# ---------------------------------------------------------------------------
# 6. Branch failure independence
# ---------------------------------------------------------------------------
print("\n--- Failure independence ---")


def raising_extract(documents, client=None):
    raise RuntimeError("simulated clinical failure")


independence_embedder = CountingEmbeddingService()
independence_components = make_components(
    index_config, independence_embedder, "workflow_independence"
)
independence = DocumentIntelligenceService(
    patient_id="case_independence",
    rag_service=RAGService(config=index_config, components=independence_components),
)
independence.ingest([FakeUpload("case.txt", "Clinical note. ECOG 1.")])

intelligence_module.extract_clinical = raising_extract
try:
    independence.run_clinical()
finally:
    intelligence_module.extract_clinical = real_extract

check(
    "Clinical failure is reported as failed",
    independence.stage_status(CLINICAL).state.value == "failed",
)
check(
    "Clinical failure does not mark RAG as failed",
    independence.stage_status(RAG).state.value == "pending",
)

rag_after_clinical = independence.ensure_rag_index()
check("RAG can still index after clinical failure", rag_after_clinical.get("success"), rag_after_clinical)
check("RAG stage succeeds independently", independence.stage_status(RAG).successful)
check(
    "Clinical stage remains failed after RAG success",
    independence.stage_status(CLINICAL).state.value == "failed",
)


def successful_extract(documents, client=None):
    return ClinicalRun(
        combined={
            "success": True,
            "data": {"patient_name": "Independent Patient"},
            "errors": [],
            "warnings": [],
        },
        per_document=(),
    )


rag_failure_components = make_components(
    index_config, CountingEmbeddingService(), "workflow_rag_failure"
)
rag_failure_service = DocumentIntelligenceService(
    patient_id="case_rag_failure",
    rag_service=RAGService(config=index_config, components=rag_failure_components),
)
rag_failure_service.ingest([FakeUpload("note.txt", "Clinical note. Stage IIA.")])

intelligence_module.extract_clinical = successful_extract
try:
    rag_failure_service.run_clinical()
finally:
    intelligence_module.extract_clinical = real_extract


def raising_index(force=False):
    raise RuntimeError("simulated indexing failure")


rag_failure_service.rag.ensure_index = raising_index
rag_failure_service.ensure_rag_index()

check("Clinical success is preserved", rag_failure_service.stage_status(CLINICAL).successful)
check(
    "RAG failure is reported as failed",
    rag_failure_service.stage_status(RAG).state.value == "failed",
)

# ---------------------------------------------------------------------------
# 7. Lazy model loading
# ---------------------------------------------------------------------------
print("\n--- Lazy loading ---")

lazy_temp = tempfile.mkdtemp(prefix="oncolink_lazy_")
try:
    lazy_config = RAGConfig()
    lazy_config.chroma_dir = lazy_temp
    lazy_config.collection_name = "workflow_lazy"

    lazy_service = RAGService(config=lazy_config)
    check("Creating RAGService builds no components", lazy_service._components is None)

    lazy_components = lazy_service.components
    check("Components are built on first use", lazy_components is not None)
    check(
        "Embedding model is still unloaded",
        not lazy_components["embedding_service"].is_loaded(),
    )
    check(
        "Reranker model is still unloaded",
        lazy_components["reranker"]._model is None,
    )

    standalone_embedder = EmbeddingService()
    standalone_reranker = Reranker(enabled=True)
    check("Standalone embedding service starts lazy", not standalone_embedder.is_loaded())
    check("Standalone reranker starts lazy", standalone_reranker._model is None)
finally:
    shutil.rmtree(lazy_temp, ignore_errors=True)

print()
print("=" * 70)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
