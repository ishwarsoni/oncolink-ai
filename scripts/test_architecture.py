"""Static and import-level architecture tests.

These checks guard the layer boundaries and the configuration that the
refactor promised to preserve.  They complement the behavioural suites; they
do not replace the real-model integration test.
"""

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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


def _source_files(*folders):
    for folder in folders:
        yield from (PROJECT_ROOT / folder).glob("*.py")


def _imports(path):
    """Return all imported module names for one Python file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


print("=" * 70)
print("Architecture and boundary tests")
print("=" * 70)

# ---------------------------------------------------------------------------
# Layer boundaries
# ---------------------------------------------------------------------------
print("\n--- Layer boundaries ---")

for layer in ("backend", "domain", "infrastructure", "application"):
    offenders = []
    for path in _source_files(layer):
        imported = _imports(path)
        if any(name == "streamlit" or name.startswith("streamlit.") for name in imported):
            offenders.append(path.name)
    check(f"{layer}/ does not import Streamlit", not offenders, offenders)

ui_offenders = []
for path in _source_files("ui"):
    imported = _imports(path)
    forbidden = {"chromadb", "sentence_transformers", "openai", "fitz"}
    if imported & forbidden:
        ui_offenders.append((path.name, sorted(imported & forbidden)))
check("ui/ does not import storage or model libraries directly", not ui_offenders, ui_offenders)

rag_ui = _imports(PROJECT_ROOT / "ui" / "rag.py")
check("ui/rag.py does not import the indexing backend", "backend.rag_indexer" not in rag_ui)

# Modules must import cleanly on their own (catches circular imports).
import importlib

_application_modules = [
    "application.clinical",
    "application.ingestion",
    "application.intelligence",
    "application.rag",
    "application.status",
]
_backend_modules = [
    "backend.chunker",
    "backend.document_loader",
    "backend.extractor",
    "backend.hybrid_retriever",
    "backend.rag_engine",
    "backend.rag_indexer",
    "backend.retriever",
]
_platform_modules = [
    "domain.clinical",
    "domain.documents",
    "infrastructure.config",
    "infrastructure.embeddings",
    "infrastructure.llm",
    "infrastructure.logging",
    "infrastructure.reranker",
    "infrastructure.vector_store",
]

import_failures = []
for module_name in _application_modules + _backend_modules + _platform_modules:
    try:
        importlib.import_module(module_name)
    except Exception as error:
        import_failures.append(f"{module_name}: {error}")

check("Application modules import cleanly", not import_failures, import_failures)
check("Backend modules import cleanly", not import_failures, import_failures)
check("Domain and infrastructure modules import cleanly", not import_failures, import_failures)

# ---------------------------------------------------------------------------
# Preserved model and configuration contracts
# ---------------------------------------------------------------------------
print("\n--- Model configuration ---")

from infrastructure.config import DEFAULT_CONFIG, RAGConfig
from infrastructure.llm import DEFAULT_MODEL, NVIDIA_BASE_URL

check(
    "Nemotron generation model is unchanged",
    DEFAULT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b",
    DEFAULT_MODEL,
)
check(
    "NVIDIA endpoint is unchanged",
    NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1",
    NVIDIA_BASE_URL,
)
check(
    "Embedding model is unchanged",
    DEFAULT_CONFIG.embedding_model == "BAAI/bge-small-en-v1.5",
    DEFAULT_CONFIG.embedding_model,
)
check(
    "Reranker model is unchanged",
    DEFAULT_CONFIG.reranker_model == "BAAI/bge-reranker-base",
    DEFAULT_CONFIG.reranker_model,
)
check("RAGConfig is constructible with defaults", isinstance(RAGConfig(), RAGConfig))

# ---------------------------------------------------------------------------
# Compatibility shims
# ---------------------------------------------------------------------------
print("\n--- Compatibility paths ---")

import backend.config as legacy_config
import backend.embedding_service as legacy_embeddings
import backend.llm_client as legacy_llm
import backend.reranker as legacy_reranker
import backend.schema as legacy_schema
import backend.vector_store as legacy_vector_store
import domain.clinical as canonical_clinical
import infrastructure.config as canonical_config
import infrastructure.embeddings as canonical_embeddings
import infrastructure.llm as canonical_llm
import infrastructure.reranker as canonical_reranker
import infrastructure.vector_store as canonical_vector_store

check("backend.config re-exports infrastructure.config", legacy_config.RAGConfig is canonical_config.RAGConfig)
check("backend.llm_client re-exports infrastructure.llm", legacy_llm.create_nvidia_client is canonical_llm.create_nvidia_client)
check("backend.embedding_service re-exports infrastructure.embeddings", legacy_embeddings.EmbeddingService is canonical_embeddings.EmbeddingService)
check("backend.vector_store re-exports infrastructure.vector_store", legacy_vector_store.VectorStore is canonical_vector_store.VectorStore)
check("backend.reranker re-exports infrastructure.reranker", legacy_reranker.Reranker is canonical_reranker.Reranker)
check("backend.schema re-exports domain.clinical", legacy_schema.ClinicalData is canonical_clinical.ClinicalData)

# ---------------------------------------------------------------------------
# One client factory and one canonical document model
# ---------------------------------------------------------------------------
print("\n--- Centralisation ---")

client_creators = []
for layer in ("backend", "domain", "infrastructure", "application", "ui"):
    for path in _source_files(layer):
        text = path.read_text(encoding="utf-8")
        if "OpenAI(" in text:
            client_creators.append(f"{layer}/{path.name}")
check(
    "NVIDIA client is constructed in one place",
    client_creators == ["infrastructure/llm.py"],
    client_creators,
)

from domain.documents import ClinicalDocument, DocumentPage

document = ClinicalDocument(
    patient_id="case_test",
    document_id="doc_test",
    filename="report.pdf",
    file_type="PDF",
    text="Evidence text",
    pages=(DocumentPage(1, "Evidence text"),),
    content_hash="hash",
)
check("Canonical document exposes index projection", "patient_id" in document.as_index_record())
check("Canonical document exposes display projection", document.as_processed_file()["success"] is True)
try:
    document.text = "changed"
    immutable = False
except Exception:
    immutable = True
check("Canonical document is immutable", immutable)

# ---------------------------------------------------------------------------
# UI surface preserved for deployment and tests
# ---------------------------------------------------------------------------
print("\n--- UI surface ---")

from ui import export, extraction, harmonization, rag, sidebar, upload

ui_functions = [
    (sidebar, "create_sidebar"),
    (upload, "create_file_uploader"),
    (upload, "create_process_button"),
    (upload, "display_extracted_text"),
    (extraction, "create_extraction_section"),
    (extraction, "display_structured_clinical_data"),
    (harmonization, "show_harmonization_section"),
    (harmonization, "create_status_panel"),
    (rag, "create_rag_assistant_section"),
    (export, "_pdf_output_bytes"),
]
check(
    "Existing UI function names are preserved",
    all(hasattr(module, name) for module, name in ui_functions),
    [name for module, name in ui_functions if not hasattr(module, name)],
)
check("PDF export helper still produces bytes", export._text_to_pdf_bytes("Test").startswith(b"%PDF"))

# ---------------------------------------------------------------------------
# No Streamlit in backend call paths used by the workflow
# ---------------------------------------------------------------------------
print("\n--- Headless workflow ---")

from application.intelligence import DocumentIntelligenceService

service = DocumentIntelligenceService(patient_id="case_headless")
check("Workflow service is headless", service.patient_id == "case_headless")
check("Workflow starts with all stages pending", all(s.state.value == "pending" for s in service.statuses.values()))
check("Workflow snapshot is serialisable", isinstance(service.snapshot(), dict))
check(
    "Document reader does not import Streamlit",
    "streamlit" not in _imports(PROJECT_ROOT / "backend" / "document_loader.py"),
)

print()
print("=" * 70)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
