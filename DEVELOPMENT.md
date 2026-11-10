# Developing OncoLink

This guide is for maintainers extending or debugging the application. It
assumes the architecture described in `ARCHITECTURE.md`.

---

## 1. Environment setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

The project is tested on Python 3.13 and the dev container uses Python 3.11.
Both are supported.

Create `.env` in the project root:

```text
NVIDIA_API_KEY=nvapi-your-key-here
```

Or configure Streamlit secrets / enter the key in the sidebar. `python-dotenv`
is loaded in `app.py` before configuration objects are built.

### Run the app

```bash
streamlit run app.py
```

The app serves on <http://localhost:8501>. Maximum upload size is 50 MB per
file (`.streamlit/config.toml`).

---

## 2. Where code belongs

| You are changing… | Put it in… |
|---|---|
| A widget, layout, label, download button | `ui/` |
| A workflow step or stage status | `application/` |
| Document identity, pages, hashes, clinical schema | `domain/` |
| A prompt, validation rule, merge rule, chunker, retriever, answer validator | `backend/` |
| NVIDIA client, embeddings, ChromaDB, reranker, config, logging | `infrastructure/` |

Rules:

1. Never import Streamlit from `backend`, `domain`, `infrastructure` or
   `application`. `scripts/test_architecture.py` enforces this.
2. Do not call `infrastructure` directly from the UI. Go through an
   application service.
3. Keep the canonical `ClinicalDocument` immutable. If a new field is needed,
   add it to the dataclass and both projections.
4. New vector-store operations must require `patient_id` and apply it as a
   filter. Never add an unscoped query.
5. Do not create a second LLM client or read `NVIDIA_API_KEY` outside
   `infrastructure/llm.py`.

### Compatibility shims

The modules `backend/config.py`, `backend/llm_client.py`,
`backend/embedding_service.py`, `backend/vector_store.py`,
`backend/reranker.py` and `backend/schema.py` exist only to keep old import
paths working. Add new code to the canonical module and, if needed, extend the
re-export list. Do not add logic to a shim.

---

## 3. Common extension tasks

### Add a supported file format

1. Add a reader in `backend/` returning text plus page metadata.
2. Add the extension to `backend.document_loader.process_uploaded_file` and its
   `_FILE_TYPES` mapping.
3. Keep PDF-style 1-based `page_number` values when the format has real pages;
   use `None` when it does not.
4. Add a regression check to `scripts/test_regression_existing.py` or the
   relevant suite.
5. Add the type to the UI uploader in `ui/upload.py` and to the README.

The canonical document layer and both branches need no changes as long as the
reader returns `{"filename", "text", "file_type", "success", "pages"}`.

### Add a clinical field

1. Add the Pydantic field and description in `domain/clinical.py`.
2. Add its display label to `FIELD_LABELS`.
3. Add it to `REQUIRED_FIELDS` only if a complete record should normally contain
   it (missing values remain warnings, not hard failures).
4. Update the extraction prompt in `backend/prompt_builder.py`.
5. Add merge behavior in `backend/harmonizer.py` if the field is not covered by
   an existing strategy.
6. Add normalization rules in `backend/normalizer.py` when needed.
7. Update `backend/summarizer.py` and the UI cards if the field should be
   displayed.
8. Add tests for validation, normalization, merge behavior and export.
9. Update the data-model table in `README.md`.

`CLINICAL_SCHEMA`, `FIELD_LABELS` and `REQUIRED_FIELDS` are derived from
`ClinicalData`, so documentation and validation cannot drift silently.

### Change an extraction prompt

Prompts live in `backend/prompt_builder.py`. Preserve these invariants:

- Treat document text as untrusted data, never as instructions.
- Return strict JSON matching `ClinicalData`.
- Use `null` for values not found; never invent facts.
- Do not add treatment recommendations.

Changing a prompt is a clinical-behavior change. Run every offline suite and,
when a key is available, the live extraction checks.

### Extend retrieval or reranking

- Chunking: `backend/chunker.py`. Keep page numbers and stable chunk IDs.
- Semantic retrieval: `backend/retriever.py`.
- Hybrid fusion / BM25 cache: `backend/hybrid_retriever.py`.
- Reranking: `infrastructure/reranker.py` plus `backend/rag_engine.py`.
- Settings: `infrastructure/config.py` (environment-overridable).

The RAG service in `application/rag.py` owns indexing and cache invalidation.
If you add an index-mutating operation, invalidate the patient's BM25 cache and
the service's signature cache.

### Add an application workflow

Put the coordination logic in `application/`, not in a UI module. Return a
result envelope or dataclass with explicit success/errors. Update the
`DocumentIntelligenceService` stage status only for the stage you own; never
mark another stage failed.

---

## 4. Configuration

All RAG settings live in `infrastructure/config.py`. See the configuration
table in `README.md`.

Frequently used variables:

```text
ONCOLINK_EMBEDDING_MODEL
ONCOLINK_EMBEDDING_DEVICE
ONCOLINK_EMBEDDING_BATCH_SIZE
ONCOLINK_CHUNK_SIZE
ONCOLINK_CHUNK_OVERLAP
ONCOLINK_CHROMA_DIR
ONCOLINK_CHROMA_COLLECTION
ONCOLINK_TOP_K
ONCOLINK_RERANK_ENABLED
ONCOLINK_RERANKER_MODEL
ONCOLINK_RAG_MODEL
NVIDIA_API_KEY
```

Boolean values accept `1`, `true`, `yes`, `on`, `y`.

The NVIDIA model (`nvidia/nemotron-3-ultra-550b-a55b`), embedding model
(`BAAI/bge-small-en-v1.5`) and reranker (`BAAI/bge-reranker-base`) are the
approved defaults. Do not change them without explicit approval and a
documented reason.

---

## 5. Testing

### Offline suite

Run from the project root:

```bash
python scripts/test_chunker.py
python scripts/test_embeddings.py
python scripts/test_vector_store.py
python scripts/test_retriever.py
python scripts/test_hybrid_retrieval.py
python scripts/test_reranker.py
python scripts/test_rag_pipeline.py
python scripts/test_patient_isolation.py
python scripts/test_regression_existing.py
python scripts/test_architecture.py
python scripts/test_workflow.py
python scripts/test_app_smoke.py
```

Expected result: **365 passed, 0 failed**. No API key, model download or
network access is required.

Legacy phase scripts are also available:

```bash
python scripts/test_phase2.py
python scripts/test_phase3.py
python scripts/test_phase4.py
python scripts/test_logging.py
```

`test_flow.py`, `test_harmonization.py` and `test_phase5.py` call the live
NVIDIA API and require `NVIDIA_API_KEY`; they are not part of the offline
suite.

### Real-model integration

```bash
# PowerShell
$env:ONCOLINK_TEST_REAL_MODELS = "1"
python scripts/test_rag_integration.py
```

This exercises the real `BAAI/bge-small-en-v1.5` model, persistent ChromaDB and
hybrid retrieval. Expected without a key: **11 passed, 0 failed**, with the
real LLM check reported as **SKIPPED**.

To include real Nemotron generation:

```powershell
$env:NVIDIA_API_KEY = "nvapi-..."
$env:ONCOLINK_TEST_REAL_MODELS = "1"
python scripts/test_rag_integration.py
```

Never report a skipped or blocked check as passed. Record the exact state.

### What the new suites cover

`scripts/test_workflow.py`:

- each upload is read exactly once
- both branches receive the same document/patient IDs
- page references survive chunking
- unchanged documents are not re-embedded
- changed documents replace stale chunks
- patient isolation and citation ownership
- clinical and RAG failures are independent
- embedding/reranker models load lazily

`scripts/test_architecture.py`:

- no Streamlit imports in lower layers
- no direct model/storage imports in the UI
- model identifiers and defaults preserved
- compatibility shims re-export the canonical objects
- only `infrastructure/llm.py` constructs the NVIDIA client
- canonical document is immutable
- UI function surface and PDF export still work

`scripts/test_app_smoke.py`:

- the real `app.py` renders via `streamlit.testing.v1.AppTest`
- extraction/RAG/extracted-text sections render with upload state
- export helpers return valid bytes

### Adding a test

Follow the existing counter style:

```python
PASS = 0
FAIL = 0

def check(name, condition, detail=""):
    ...
```

Return a non-zero exit code when `FAIL` is non-zero. Do not weaken an existing
check or replace a real integration check with a mock. Use deterministic fakes
for external services only in the offline suites.

---

## 6. Debugging

### Logs

`infrastructure/logging.py` writes to `logs/application.log` using a rotating
handler (5 MB, 3 backups). Logging is initialised once per process; Streamlit
reruns do not duplicate handlers.

Logs contain event descriptions and error messages, never API keys. Raw user
questions are not logged by default.

### Common issues

**“NVIDIA_API_KEY not found”**

Set `.env`, Streamlit secrets, an environment variable, or the sidebar field.
The placeholder `nvapi-your-key-here` is rejected.

**ChromaDB errors**

Stop the app, move `storage/chroma_db` aside, and rebuild the index. A changed
document replaces its own chunks; deleting the whole directory resets all
cases.

**Embedding model will not download**

Check network/disk. Set `HF_TOKEN` for rate limits or
`ONCOLINK_EMBEDDING_MODEL` for an alternate model. Offline: pre-download and set
`HF_HUB_OFFLINE=1`.

**RAG answers fail but the index exists**

Answer generation needs a key; indexing does not. Check the sidebar status and
`infrastructure/llm.py` errors in the log.

**A cited page looks wrong**

Check the source PDF has a text layer and that `DocumentPage.page_number` is
populated. DOCX/TXT intentionally report “Page not available”.

---

## 7. Code style

- Python 3.11+ type hints; prefer built-in generics (`list[str]`, `dict[str, Any]`).
- Module/class/function docstrings explain intent, not mechanics.
- Comments explain *why* a clinical rule, security boundary or cache exists.
- Keep functions focused; avoid deeply nested conditionals.
- Do not add an abstraction unless it removes real duplication or enforces a
  boundary.
- Run `python -m pyflakes backend domain infrastructure application ui app.py`
  before committing.
- Formatting is Black-compatible; run `black .` if installed.

### Definition of done

1. The change is in the correct layer.
2. Offline suites still pass (365 checks).
3. New behavior has a regression check.
4. Documentation (`README.md`, `ARCHITECTURE.md` if boundaries changed) is updated.
5. No secrets, logs, vector data or `.env` files are staged.
6. If a live check could not run, it is reported as SKIPPED/BLOCKED, never PASSED.
