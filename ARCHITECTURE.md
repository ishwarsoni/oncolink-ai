# OncoLink Architecture

This document describes the implementation that ships in this repository. It is
the reference for where new code belongs and why the boundaries exist.

> OncoLink is educational/demonstration software, not a medical device. See the
> license and disclaimer in `README.md`.

---

## 1. Goals

The architecture has one job: let two different clinical capabilities consume
the same documents without duplicating reads, hashing, embeddings or state.

- **One canonical representation** — `domain.documents.ClinicalDocument` is
  built once per upload and passed to both branches.
- **Two independent capabilities** — structured clinical extraction and
  evidence retrieval/RAG are never merged into one LLM call.
- **Independent failure domains** — a clinical failure must not corrupt the RAG
  index or mark it failed; a RAG failure must not discard valid clinical data.
- **Strong patient isolation** — every storage boundary requires a patient/case
  ID and validates it again when rows are read back.
- **Lazy, bounded resource use** — models load on first use; document extraction
  uses a bounded worker pool; unchanged documents are not re-embedded.
- **Framework-free core** — `backend`, `domain`, `infrastructure` and
  `application` contain no Streamlit imports.

---

## 2. Layers

```text
app.py
  └── ui/                 Streamlit rendering, session state, exports
        └── application/  workflow coordination and per-stage status
              ├── domain/          models and core rules (framework-free)
              ├── backend/         clinical / retrieval algorithms
              └── infrastructure/  NVIDIA, ChromaDB, models, config, logging
```

Dependency rule: arrows point downward only. `ui` may import `application`;
`application` may import `domain`, `backend` and `infrastructure`. No lower
layer imports `ui`, and `backend`/`domain`/`infrastructure`/`application` never
import Streamlit.

| Layer | Modules | Owns |
|---|---|---|
| Entry point | `app.py` | Environment loading, page order, wiring. |
| Presentation | `ui/` | Widgets, session state, service cache, rendering, downloads. |
| Application | `application/` | Ingestion, clinical/RAG workflows, stage statuses, lazy RAG lifecycle. |
| Domain | `domain/` | `ClinicalDocument`, `DocumentPage`, IDs/hashes/dates, `ClinicalData` schema. |
| Backend | `backend/` | Readers, prompts, validation, normalization, harmonization, conflicts, chunking, retrieval, answer validation. |
| Infrastructure | `infrastructure/` | NVIDIA client, config, embeddings, ChromaDB, reranker, logging. |

### Compatibility paths

The pre-refactor application exposed configuration, schema, LLM, embedding,
vector-store and reranker modules under `backend/`. Those files now contain thin
re-exports so existing integrations and the regression suite keep working:

| Legacy path | Canonical module |
|---|---|
| `backend/config.py` | `infrastructure/config.py` |
| `backend/llm_client.py` | `infrastructure/llm.py` |
| `backend/embedding_service.py` | `infrastructure/embeddings.py` |
| `backend/vector_store.py` | `infrastructure/vector_store.py` |
| `backend/reranker.py` | `infrastructure/reranker.py` |
| `backend/schema.py` | `domain/clinical.py` |

New code should import the canonical module. The shims contain no logic.

---

## 3. Canonical document model

`domain/documents.py` defines immutable `ClinicalDocument` and `DocumentPage`
objects. Both branches receive the same instances.

A `ClinicalDocument` carries:

| Field | Meaning |
|---|---|
| `patient_id` | Random, non-identifying case ID (`case_<12 hex>`). |
| `document_id` | Stable `sha256(patient_id \| filename)[:16]`. |
| `filename` | Sanitised display/citation name. |
| `file_type` | `PDF`, `DOCX`, `TXT`, or `UNKNOWN`. |
| `text` | Full extracted text, read once. |
| `pages` | Tuple of 1-based `DocumentPage` values (or `None` for DOCX/TXT). |
| `content_hash` | SHA-256 of `text`; drives incremental indexing. |
| `upload_timestamp` | UTC ingestion timestamp. |
| `document_date` | Explicit labelled date parsed from the text, ISO when possible. |

Two projections keep the boundary explicit:

- `as_index_record()` → dictionary consumed by the chunker and vector store.
- `as_processed_file()` → legacy loader shape consumed by UI display.

`index_signature(documents)` hashes sorted `document_id:content_hash` pairs.
`application.rag.RAGService` uses it to invalidate its index cache when the
document set changes.

The document is frozen so one branch cannot mutate text or identifiers the
other branch is using.

---

## 4. Ingestion

`application/ingestion.py::ingest_uploads` is the only place an upload is read.

```text
UploadedFile
  → backend.document_loader.process_uploaded_file   (one read)
  → display record (success/failure, pages, raw text)
  → ClinicalDocument.from_processed_file            (canonical, hashed)
  → IngestionResult(documents, display_records, errors, warnings)
```

Properties:

- A reader exception becomes a failed display record; other files still load.
- Empty files stay visible in the UI but never become canonical documents.
- Duplicate document IDs (same patient + filename) are indexed only once.
- Filenames are sanitised against path traversal and control characters.
- PDF page numbers are preserved 1-based. DOCX/TXT pages are `None` rather than
  invented.

`DocumentIntelligenceService.ingest` stores the result, attaches the documents
to the RAG service, and resets downstream results because the document set has
changed. This is a lifecycle reset, not a branch failure.

---

## 5. Structured clinical intelligence

`application/clinical.py` coordinates two extraction passes over the same
canonical documents:

1. **Combined pass** — all documents concatenated with document markers, giving
   the model cross-document context. This powers the main structured view.
2. **Per-document pass** — one extraction per document, returning results in
   upload order. This preserves provenance for harmonization and conflict
   detection.

A single NVIDIA client is created per workflow and shared by both passes and by
the bounded worker pool (`max_workers = 5`). Each document result is isolated;
one failed extraction becomes a per-document error, not a crash.

```text
ClinicalDocument(s)
  → backend.extractor.extract_from_multiple_documents
  → backend.extractor.extract_each_document
  → domain.clinical.ClinicalData validation + backend.normalizer
  → backend.harmonizer.harmonize_extractions
  → backend.conflict_detector.detect_conflicts
  → backend.summarizer.generate_patient_summary
```

Harmonization order is deterministic: the per-document list preserves upload
order and the merge strategies operate in a fixed field order. Conflicting
values are reported instead of silently resolved.

Failure semantics: a combined extraction failure leaves `data = None` and
records errors. The orchestrator marks only the clinical stage failed. It never
synthesizes empty clinical data to make the stage look successful.

---

## 6. Evidence retrieval and RAG

`application/rag.py::RAGService` owns the lazy RAG lifecycle for one case.

```text
ClinicalDocument(s)
  → backend.chunker.chunk_document            (1-based page metadata)
  → infrastructure.embeddings                 (BAAI/bge-small-en-v1.5)
  → infrastructure.vector_store.upsert_chunks (patient-scoped ChromaDB)
```

Indexing is incremental:

- A document whose stored `content_hash` matches is skipped.
- A changed document has all previous chunks deleted before the new version is
  written, so stale evidence can never be retrieved.
- The BM25 cache is invalidated for the patient after any index change.
- A session-level signature cache avoids repeating index bookkeeping on every
  Streamlit rerun, but never hides a changed document set.

Question flow:

```text
question
  → hybrid retrieval (semantic + BM25 → RRF) or semantic only
  → optional cross-encoder rerank (BAAI/bge-reranker-base)
  → backend.rag_engine prompt with evidence IDs
  → infrastructure.llm.call_llm (Nemotron)
  → JSON parse + answer validation
  → citation_manager rebuilds filenames/pages from stored chunk metadata
```

Reranking is chosen per question and the model loads only when first enabled.
If the reranker cannot load, the original ranking is returned with a warning.

### Lazy components

`RAGService.components` builds the component bundle on first access. Creating
the service, ingesting documents, or rendering the page does not download a
model. `EmbeddingService` and `Reranker` each cache their model per process.

### Patient isolation in the RAG path

- Every vector-store read/write requires `patient_id` and applies it as a
  metadata filter.
- Returned rows are checked again; mismatched rows are discarded.
- `RAGService.set_documents` refuses to index a mixed-patient batch.
- Retrieval, prompt construction and citation building are all scoped to one
  patient.
- Citation filenames and pages are taken from stored chunk metadata, never from
  model output. A citation that does not match a retrieved evidence ID is
  dropped.

---

## 7. Orchestration and statuses

`application/intelligence.py::DocumentIntelligenceService` is the single facade
the UI uses.

It exposes:

- `ingest(uploaded_files)` → `IngestionResult`
- `run_clinical()` → `ClinicalRun`
- `harmonize()` → `ClinicalSummary`
- `ensure_rag_index(force=False)` → index result
- `ask(question, ...)` → cited answer
- `index_status()` / `delete_rag_index()`
- `stage_status(stage)` / `snapshot()`
- `reset()` for a new case

Each stage has an independent `StageStatus` (`application/status.py`):

| State | Meaning |
|---|---|
| `pending` | Not run yet. |
| `running` | Started; no result yet. |
| `succeeded` | Completed with a usable result. |
| `failed` | Failed; errors are attached. |
| `skipped` | Did not run (no documents, deleted index, etc.). |

Failure independence is explicit:

- `run_clinical` catches extraction exceptions, records a failed clinical
  status, and leaves the RAG status untouched.
- `ensure_rag_index` / `ask` record RAG failures and leave clinical data and
  clinical status untouched.
- A later successful RAG operation does not retroactively mark a failed
  clinical stage as successful.

---

## 8. Data flow summary

```text
                   ┌─────────────────────────────┐
Uploaded files ───►│ application.ingestion       │
                   │ one read → ClinicalDocument │
                   └──────────────┬──────────────┘
                                  │ shared immutable documents
                 ┌────────────────┴─────────────────┐
                 ▼                                  ▼
   application.clinical                 application.rag
   extraction + harmonization            chunk + embed + index
                 │                                  │
                 ▼                                  ▼
   ClinicalRun / ClinicalSummary         cited answer / evidence
                 └────────────────┬─────────────────┘
                                  ▼
                   UI reads results from session state
```

`ui/session.py::get_intelligence_service` caches the service per patient case
and rebuilds it when the case ID changes.

The UI presents three workspace views over that one cached service: Patient
Record reads extraction/harmonization results from session state, AI Clinical
Chat calls `service.ask` against the patient-scoped index, and Uploaded
Documents renders the canonical display records. View switching does not
re-ingest documents or reset either stage status.

---

## 9. Configuration and model management

| Concern | Location | Default |
|---|---|---|
| Generation model | `infrastructure/llm.py` | `nvidia/nemotron-3-ultra-550b-a55b` |
| NVIDIA endpoint | `infrastructure/llm.py` | `https://integrate.api.nvidia.com/v1` |
| Embeddings | `infrastructure/config.py` | `BAAI/bge-small-en-v1.5` |
| Reranker | `infrastructure/config.py` | `BAAI/bge-reranker-base` |
| RAG knobs | `infrastructure/config.py` | env-overridable |
| API key | `ui/session.py` → env → `infrastructure/llm.py` | none |
| Logging | `infrastructure/logging.py` | rotating 5 MB × 3 |

`infrastructure/llm.py` is the only module that reads `NVIDIA_API_KEY` or
constructs an OpenAI client. The UI resolves Streamlit secrets and the sidebar
field into that environment variable before any pipeline runs. The key is never
logged.

The model identifiers and clinical schema/prompts are preserved from the
pre-refactor application.

---

## 10. Error handling

The code uses two complementary styles:

- **Workflow results** — user-facing operations return dictionaries/dataclasses
  with `success`, `data`/`errors`, and `warnings`. This matches the legacy
  result envelopes and lets the UI render partial success.
- **Stage status** — the orchestrator records independent stage state for the
  UI and diagnostics.

Expected failures (unreadable file, malformed LLM JSON, missing API key, empty
retrieval) are reported. Unexpected exceptions inside one stage are caught at
the stage boundary, logged, and converted into that stage's failure. The other
stage and the application stay usable.

---

## 11. Testing architecture

The offline suite has three kinds of checks:

1. **Behavioral suites** (`test_chunker.py` … `test_patient_isolation.py`) —
   real algorithms with deterministic fake embeddings/LLM where external
   services would otherwise be required.
2. **New workflow/architecture suites** (`test_workflow.py`,
   `test_architecture.py`) — single-read ingestion, shared IDs/pages,
   incremental indexing, branch independence, lazy loading, layer boundaries,
   model configuration and UI surface.
3. **AppTest smoke test** (`test_app_smoke.py`) — executes the real `app.py`
   and checks that the page and sections render without exceptions.

The optional real-model integration test (`test_rag_integration.py`) uses the
real embedding model and ChromaDB. It reports the NVIDIA generation check as
**SKIPPED** when no key is available; it is never counted as passed.

See `DEVELOPMENT.md` for commands and contribution rules.

---

## 12. Deployment compatibility

- The entry point remains `app.py`; `streamlit run app.py` is unchanged.
- Streamlit secrets/environment/sidebar key resolution is unchanged.
- The ChromaDB directory and collection defaults are unchanged, so an existing
  index remains readable.
- Exports remain PDF, TXT fallback and JSON views.
- Legacy `backend.*` import paths still resolve through compatibility shims.
- No new third-party dependency is required for the refactor.

Limitations remain: no authentication, no OCR, no clinical validation, and no
guarantee that a local Chroma directory survives an ephemeral deployment.
