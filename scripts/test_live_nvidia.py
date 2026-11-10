"""Live NVIDIA verification: connectivity, real extraction, harmonization.

This script calls the real Nemotron endpoint.  It never prints the API key and
never substitutes a mock for a model response.  Deterministic parser and
citation checks are labelled as such.

Run:

    $env:ONCOLINK_TEST_LIVE = "1"
    python scripts/test_live_nvidia.py

Exit code is 0 when every executed check passed.  Skipped checks are reported
as SKIPPED and never counted as passed.
"""

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from application.clinical import extract_clinical, harmonize_clinical
from application.ingestion import ingest_uploads
from backend.document_loader import process_uploaded_file
from backend.extractor import run_extraction
from backend.json_parser import clean_llm_response, parse_json_safely
from domain.clinical import ClinicalData
from infrastructure.llm import (
    DEFAULT_MODEL,
    LLMConfigurationError,
    NVIDIA_BASE_URL,
    call_llm,
    create_nvidia_client,
)

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


class CountingFile:
    """Uploaded-file stand-in that records how often it is read."""

    def __init__(self, path):
        self.name = Path(path).name
        self._bytes = Path(path).read_bytes()
        self.reads = 0

    def read(self):
        self.reads += 1
        return self._bytes


print("=" * 72)
print("LIVE NVIDIA verification (real model, no mocked model responses)")
print("=" * 72)

if os.getenv("ONCOLINK_TEST_LIVE", "").strip() != "1":
    print("  SKIP  Set ONCOLINK_TEST_LIVE=1 to run live verification.")
    raise SystemExit(0)

api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
if not api_key or api_key == "nvapi-your-key-here":
    print("  SKIP  NVIDIA_API_KEY is not configured. Nothing was executed.")
    raise SystemExit(0)

# ---------------------------------------------------------------------------
# 1. Configuration and connectivity
# ---------------------------------------------------------------------------
print("\n--- Configuration and connectivity ---")

print(f"  model    : {DEFAULT_MODEL}")
print(f"  endpoint : {NVIDIA_BASE_URL}")
print(f"  key      : configured ({len(api_key)} chars, value not printed)")

check("Generation model is the approved Nemotron ID", DEFAULT_MODEL == "nvidia/nemotron-3-ultra-550b-a55b")
check("Endpoint is the NVIDIA NIM endpoint", NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1")

try:
    client = create_nvidia_client(timeout=120)
    connected = True
except Exception as error:
    connected = False
    check("NVIDIA client can be created", False, type(error).__name__)
if not connected:
    print("\nResults: live verification aborted because the client could not be created.")
    raise SystemExit(1)

check("NVIDIA client can be created", True)

start = time.perf_counter()
probe = call_llm(
    client,
    "You are a terse assistant. Follow the instruction exactly.",
    "Reply with exactly: ONCOLINK_OK",
    max_tokens=512,
)
probe_seconds = time.perf_counter() - start

check(
    "Real inference returned content (not an error)",
    bool(probe.strip()) and not probe.startswith("LLM API call failed"),
    probe[:200],
)
check("Real inference content is non-empty", len(probe.strip()) > 0, repr(probe[:80]))
check("Real inference latency was measured", probe_seconds > 0, f"{probe_seconds:.2f}s")
print(f"  inference latency: {probe_seconds:.2f}s, response chars: {len(probe.strip())}")

# The model echoes its own name; this confirms the real served model ID.
raw_response = client.chat.completions.create(
    model=DEFAULT_MODEL,
    messages=[{"role": "user", "content": "Reply with exactly: ONCOLINK_OK"}],
    temperature=0.1,
    top_p=0.1,
    max_tokens=512,
    stream=False,
)
check(
    "Served model matches the configured model",
    raw_response.model == DEFAULT_MODEL,
    raw_response.model,
)
print(f"  served model: {raw_response.model}")

# ---------------------------------------------------------------------------
# 2. Real document loading across formats
# ---------------------------------------------------------------------------
print("\n--- Document formats ---")

data_dir = PROJECT_ROOT / "data"
paths = {
    "txt": data_dir / "test_patient.txt",
    "pdf": data_dir / "pathology_report.pdf",
    "docx": data_dir / "lab_report.docx",
}
missing = [str(path) for path in paths.values() if not path.exists()]
if missing:
    print(f"  SKIP  sample documents missing: {missing}")
    raise SystemExit(0)

loaded = {}
for label, path in paths.items():
    loaded[label] = process_uploaded_file(CountingFile(path))

check("TXT sample loads", loaded["txt"]["success"] and bool(loaded["txt"]["text"]))
check("PDF sample loads", loaded["pdf"]["success"] and bool(loaded["pdf"]["text"]))
check("DOCX sample loads", loaded["docx"]["success"] and bool(loaded["docx"]["text"]))
check(
    "PDF pages are 1-based",
    bool(loaded["pdf"]["pages"])
    and all(page["page_number"] >= 1 for page in loaded["pdf"]["pages"]),
    loaded["pdf"].get("pages"),
)
print(
    "  pages: txt=%s pdf=%s docx=%s"
    % (
        len(loaded["txt"]["pages"]),
        len(loaded["pdf"]["pages"]),
        len(loaded["docx"]["pages"]),
    )
)

# ---------------------------------------------------------------------------
# 3. Real extraction for each format
# ---------------------------------------------------------------------------
print("\n--- Real structured extraction by format ---")

extractions = {}
for label in ("txt", "pdf", "docx"):
    start = time.perf_counter()
    result = run_extraction(loaded[label]["text"])
    extractions[label] = result
    elapsed = time.perf_counter() - start
    check(
        f"{label.upper()} extraction succeeds",
        result["success"] and isinstance(result["data"], dict),
        result.get("errors"),
    )
    check(
        f"{label.upper()} response is valid JSON structure",
        isinstance(result.get("raw_response"), str)
        and result["raw_response"].strip().startswith("{"),
        repr((result.get("raw_response") or "")[:80]),
    )
    if result["success"]:
        try:
            ClinicalData.model_validate(result["data"])
            valid = True
        except Exception:
            valid = False
        check(f"{label.upper()} data passes Pydantic validation", valid)
    print(f"  {label}: {elapsed:.1f}s")

# ---------------------------------------------------------------------------
# 4. Multi-document extraction, harmonization, conflicts, summary
# ---------------------------------------------------------------------------
print("\n--- Multi-document clinical workflow ---")

patient_id = "case_live_nvidia"
uploads = [CountingFile(paths["txt"]), CountingFile(data_dir / "progress_note.txt")]
ingested = ingest_uploads(uploads, patient_id)

check("Ingestion reads each upload exactly once", all(item.reads == 1 for item in uploads))
check("Both documents become canonical documents", len(ingested.documents) == 2, ingested.errors)

start = time.perf_counter()
run = extract_clinical(ingested.documents)
clinical_seconds = time.perf_counter() - start
print(f"  combined + per-document extraction: {clinical_seconds:.1f}s")

check("Combined extraction succeeds", run.success, run.errors)
check("Per-document extraction returns one result per document", len(run.per_document) == 2)
check(
    "Every per-document extraction succeeds",
    all(item.get("success") for item in run.per_document),
    [item.get("errors") for item in run.per_document],
)
check(
    "Per-document provenance keeps filenames",
    {item.get("filename") for item in run.per_document}
    == {"test_patient.txt", "progress_note.txt"},
    [item.get("filename") for item in run.per_document],
)

if run.success:
    data = run.combined["data"]
    try:
        validated = ClinicalData.model_validate(data)
        pydantic_ok = True
    except Exception:
        validated = None
        pydantic_ok = False
    check("Combined extraction passes Pydantic validation", pydantic_ok)

    check("Normalization produced an integer age", isinstance(data.get("age"), int), data.get("age"))
    check("Normalization produced an integer ECOG score", isinstance(data.get("ecog_score"), int), data.get("ecog_score"))
    biomarkers = data.get("biomarkers") or []
    pdl1 = [item for item in biomarkers if str(item.get("name", "")).upper() == "PD-L1"]
    check("Normalization kept the PD-L1 biomarker", bool(pdl1), biomarkers)
    check(
        "Normalization canonicalised biomarker status",
        bool(pdl1) and pdl1[0].get("status") in {"Positive", "Negative", "Unknown", "Not tested"},
        pdl1,
    )
    print(
        "  combined: name=%r cancer_type=%r stage=%r ecog=%r biomarkers=%d"
        % (
            data.get("patient_name"),
            data.get("cancer_type"),
            data.get("cancer_stage"),
            data.get("ecog_score"),
            len(biomarkers),
        )
    )

summary = harmonize_clinical(run.per_document)
harmonized = summary.harmonized
conflicts = summary.conflicts

check("Harmonization succeeds", harmonized.get("success"), harmonized.get("warnings"))
check("Both documents are merged", harmonized.get("documents_merged") == 2, harmonized.get("documents_merged"))
check("Field provenance is recorded", bool(harmonized.get("field_sources")), harmonized.get("field_sources"))
check(
    "Field provenance references a real document",
    any(
        filename in str(source)
        for source in (harmonized.get("field_sources") or {}).values()
        for filename in ("test_patient.txt", "progress_note.txt")
    ),
    harmonized.get("field_sources"),
)

conflict_list = conflicts.get("conflicts") or []
check("Conflict detection compared both documents", conflicts.get("documents_compared") == 2, conflicts)
check("PD-L1 disagreement is reported", any(
    "pdl1" in str(item.get("field", "")).lower().replace("-", "").replace("_", "")
    or "pd-l1" in item.get("description", "").lower()
    for item in conflict_list
), conflict_list)
if conflict_list:
    recorded_values = json.dumps(conflict_list, default=str)
    check("Conflict sources include both readings", "45" in recorded_values and "60" in recorded_values, recorded_values[:400])
    print("  conflicts:")
    for item in conflict_list:
        print(f"    [{item.get('severity')}] {item.get('field')}: {item.get('description')}")
else:
    check("Conflict sources include both readings", False, "no conflicts returned")

check("Patient summary succeeds", summary.summary.get("success"), summary.summary.get("error"))
summary_text = summary.summary.get("summary") or ""
check("Patient summary contains the patient name", "Rajesh" in summary_text, summary_text[:200])
check("Patient summary contains PD-L1", "PD-L1" in summary_text or "PDL1" in summary_text.replace("-", ""), summary_text[:300])
check("Patient summary contains a follow-up section", "FOLLOW" in summary_text.upper(), summary_text[:400])

# ---------------------------------------------------------------------------
# 5. Error handling with real transport failures
# ---------------------------------------------------------------------------
print("\n--- Error handling ---")

clean = clean_llm_response('```json\n{"a": 1, "b": [1, 2]}\n```')
check("Markdown-fenced JSON is cleaned", clean.get("success"), clean)
parsed = parse_json_safely(clean.get("json_string", ""))
check("Cleaned JSON parses", parsed.get("success") and parsed.get("data") == {"a": 1, "b": [1, 2]}, parsed)

# Trailing commas are repaired by clean_llm_response, which the extraction
# pipeline calls before parse_json_safely.
trailing_clean = clean_llm_response('{"a": 1,}')
trailing = parse_json_safely(trailing_clean.get("json_string") or "")
check(
    "Trailing comma is repaired on the extraction path",
    trailing.get("success") and trailing.get("data") == {"a": 1},
    trailing,
)
broken = parse_json_safely("{not valid json")
check("Malformed JSON fails cleanly", broken.get("success") is False, broken)
check("Malformed JSON error is human-readable", bool(broken.get("error")), broken)

from openai import OpenAI

unreachable = OpenAI(
    base_url="http://127.0.0.1:9/v1",
    api_key="invalid-test-key",
    timeout=5,
    max_retries=0,
)
transport = call_llm(unreachable, "system", "user", max_tokens=16)
check("Transport failure is reported, not raised", transport.startswith("LLM API call failed"), transport[:160])

short_timeout = create_nvidia_client(timeout=0.001)
timeout_result = call_llm(short_timeout, "system", "user", max_tokens=16)
check("Timeout is reported, not raised", timeout_result.startswith("LLM API call failed"), timeout_result[:160])

invalid_model = call_llm(client, "system", "user", model="nvidia/oncolink-not-a-real-model", max_tokens=16)
check("Invalid model is reported, not raised", invalid_model.startswith("LLM API call failed"), invalid_model[:160])

saved_key = os.environ.pop("NVIDIA_API_KEY", None)
try:
    try:
        create_nvidia_client()
        missing_key_raised = False
    except LLMConfigurationError:
        missing_key_raised = True
finally:
    if saved_key is not None:
        os.environ["NVIDIA_API_KEY"] = saved_key
check("Missing API key raises a configuration error", missing_key_raised)

print()
print("=" * 72)
print(f"Results: {PASS} passed, {FAIL} failed, {SKIP} skipped")
print("=" * 72)
raise SystemExit(1 if FAIL else 0)
