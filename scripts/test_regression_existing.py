"""Regression tests for the pre-existing structured extraction modules.

These checks do not call the LLM.  They verify that the original
document-loading, validation, normalization, harmonization, conflict
detection, summary, and export building blocks still work after the RAG
integration.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
print("Existing structured-pipeline regression tests")
print("=" * 70)

# ---- Imports -----------------------------------------------------------
try:
    from backend.document_loader import process_uploaded_file
    from backend.json_parser import clean_llm_response, parse_json_safely
    from backend.schema import ClinicalData
    from backend.validator import validate_clinical_data
    from backend.normalizer import run_normalization_pipeline
    from backend.harmonizer import harmonize_extractions
    from backend.conflict_detector import detect_conflicts
    from backend.summarizer import generate_patient_summary
    import backend.extractor  # noqa: F401
    import backend.prompt_builder  # noqa: F401
    import backend.llm_client  # noqa: F401
    import backend.merge_utils

    _ = backend.merge_utils  # imported only to confirm the module loads

    check("All existing backend modules import", True)
except Exception as error:
    check("All existing backend modules import", False, repr(error))
    raise SystemExit(1)

# ---- JSON parser -------------------------------------------------------
fenced = "```json\n{\"answer\": \"ok\"}\n```"
cleaned = clean_llm_response(fenced)
check("Fenced JSON is cleaned", cleaned.get("success") and "\"answer\"" in cleaned.get("json_string", ""), cleaned)

trailing = clean_llm_response('{"patient_name": "Test", "age": 58,}')
parsed = parse_json_safely(trailing.get("json_string") or "")
check("Trailing comma JSON is cleaned and parsed", trailing.get("success") and parsed.get("success"), parsed)

bad = parse_json_safely("not json at all")
check("Invalid JSON is reported safely", bad.get("success") is False and bool(bad.get("error")))

# ---- Schema / validator -----------------------------------------------
good_data = {
    "patient_name": "Test Patient",
    "age": 58,
    "gender": "Female",
    "diagnosis": "Non-small cell lung cancer",
    "cancer_type": "NSCLC",
    "cancer_stage": "IIIA",
    "ecog_score": 1,
    "biomarkers": [
        {"name": "PD-L1", "value": "45%", "status": "Positive"},
        {"name": "EGFR", "value": "Positive", "status": "Exon 19 deletion"},
    ],
    "current_medication": "Pembrolizumab",
    "previous_treatment": "Carboplatin and pemetrexed",
    "follow_up_plan": "Repeat imaging in three months",
}
model = ClinicalData(**good_data)
check("Pydantic schema accepts valid data", model.patient_name == "Test Patient")

validation = validate_clinical_data(good_data)
check("Validator accepts valid data", validation.get("valid") is True, validation)
check("Validator returns data", isinstance(validation.get("data"), dict))

invalid = validate_clinical_data({"age": 500, "ecog_score": 9})
check("Validator rejects out-of-range values", invalid.get("valid") is False, invalid)
check("Validator reports errors", bool(invalid.get("errors")))

# ---- Normalizer --------------------------------------------------------
raw = dict(good_data)
raw["patient_name"] = "  test patient  "
raw["gender"] = "female"
raw["cancer_stage"] = " iiia "
normalized = run_normalization_pipeline(raw)
check("Normalization succeeds", normalized.get("success"), normalized)
check("Name is trimmed and title-cased", normalized["data"]["patient_name"] == "Test Patient", normalized["data"].get("patient_name"))
check("Normalization does not mutate the input", raw["patient_name"] == "  test patient  ")

# ---- Document loading --------------------------------------------------
data_dir = PROJECT_ROOT / "data"


class MockFile:
    def __init__(self, path):
        self.name = Path(path).name
        self._bytes = Path(path).read_bytes()

    def read(self):
        return self._bytes


for filename, expected_type in [
    ("test_patient.txt", "TXT"),
    ("lab_report.docx", "DOCX"),
    ("pathology_report.pdf", "PDF"),
]:
    path = data_dir / filename
    if not path.exists():
        check(f"{expected_type} sample exists", False, str(path))
        continue
    loaded = process_uploaded_file(MockFile(path))
    check(f"{expected_type} document loads", loaded["success"], loaded.get("text", "")[:120])
    check(f"{expected_type} document has text", len(loaded.get("text", "")) > 0)
    check(f"{expected_type} document includes pages metadata", isinstance(loaded.get("pages"), list))

pdf_loaded = process_uploaded_file(MockFile(data_dir / "pathology_report.pdf"))
check("PDF pages are 1-based", all(page["page_number"] >= 1 for page in pdf_loaded["pages"]), pdf_loaded["pages"])

# ---- Harmonization -----------------------------------------------------
def extraction(filename, data):
    return {"success": True, "data": data, "errors": [], "warnings": [], "filename": filename}


doc_one = extraction("first.txt", {
    "patient_name": "Test Patient",
    "age": 58,
    "gender": "Female",
    "diagnosis": "Non-small cell lung cancer",
    "cancer_stage": "IIIA",
    "biomarkers": [{"name": "PD-L1", "value": "45%", "status": "Positive"}],
    "current_medication": "Pembrolizumab",
    "ecog_score": 1,
})
doc_two = extraction("second.txt", {
    "patient_name": "Test Patient",
    "age": 58,
    "gender": "Female",
    "diagnosis": "Non-small cell lung cancer with mediastinal involvement",
    "cancer_stage": "IIIB",
    "biomarkers": [
        {"name": "PD-L1", "value": "60%", "status": "Positive"},
        {"name": "EGFR", "value": "Positive", "status": "Exon 19 deletion"},
    ],
    "previous_treatment": "Carboplatin and pemetrexed",
    "ecog_score": 2,
})
failed = {"success": False, "data": None, "errors": ["boom"], "warnings": [], "filename": "failed.txt"}

harmonized = harmonize_extractions([doc_one, doc_two, failed])
check("Harmonization succeeds", harmonized["success"], harmonized)
check("Harmonization merges valid documents only", harmonized["documents_merged"] == 2, harmonized)
check("Harmonization tracks field sources", isinstance(harmonized.get("field_sources"), dict) and bool(harmonized["field_sources"]))
check("Harmonization keeps the longest diagnosis", "mediastinal" in harmonized["data"]["diagnosis"], harmonized["data"]["diagnosis"])
check("Harmonization merges biomarkers", {bm["name"] for bm in harmonized["data"]["biomarkers"]} == {"PD-L1", "EGFR"}, harmonized["data"]["biomarkers"])
check("Harmonization combines treatments", "Pembrolizumab" in (harmonized["data"].get("current_medication") or ""))
check("Harmonization combines previous treatment", "Carboplatin" in (harmonized["data"].get("previous_treatment") or ""))

no_data = harmonize_extractions([failed])
check("Harmonization handles no valid data", no_data["success"] is False and no_data["data"] is None)

# ---- Conflict detection ------------------------------------------------
conflicts = detect_conflicts([doc_one, doc_two])
check("Conflict detection succeeds", conflicts["success"], conflicts)
check("Conflict detection compares documents", conflicts["documents_compared"] == 2)
fields = {conflict["field"] for conflict in conflicts["conflicts"]}
check("Stage conflict detected", "cancer_stage" in fields, fields)
check("ECOG conflict detected", "ecog_score" in fields, fields)
check("Biomarker conflict detected", any(field.startswith("biomarker") for field in fields), fields)
check("Conflicts include source documents", all(conflict.get("sources") for conflict in conflicts["conflicts"]))

no_conflicts = detect_conflicts([doc_one])
check("Single document has no conflicts", no_conflicts["success"] and no_conflicts["conflicts"] == [])

# ---- Summary -----------------------------------------------------------
summary = generate_patient_summary(harmonized["data"], harmonized.get("field_sources"), conflicts.get("conflicts", []))
check("Summary succeeds", summary.get("success"), summary)
check("Summary is non-empty text", isinstance(summary.get("summary"), str) and len(summary["summary"]) > 0)
check("Summary mentions the patient", "Test Patient" in summary["summary"])
check("Summary flags conflicts", "CONFLICTS DETECTED" in summary["summary"])

# ---- Export building blocks -------------------------------------------
serialized = json.dumps(harmonized["data"], indent=2)
check("Harmonized record serializes to JSON", json.loads(serialized)["patient_name"] == "Test Patient")

try:
    from fpdf import FPDF

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Courier", "", 9)
    for line in serialized.split("\n"):
        pdf.cell(0, 4.5, line.encode("latin-1", errors="replace").decode("latin-1"), new_x="LMARGIN", new_y="NEXT")
    pdf_bytes = pdf.output()
    if isinstance(pdf_bytes, str):
        pdf_bytes = pdf_bytes.encode("latin-1")
    pdf_bytes = bytes(pdf_bytes)
    check("PDF export builds a PDF", pdf_bytes.startswith(b"%PDF") and len(pdf_bytes) > 100)
except Exception as error:
    check("PDF export builds a PDF", False, repr(error))

# ---- App integration surface ------------------------------------------
try:
    import app
    from ui import export, extraction, harmonization, rag, sidebar, upload

    surfaces = (
        ("app", app, "main"),
        ("ui.sidebar", sidebar, "create_sidebar"),
        ("ui.upload", upload, "create_file_uploader"),
        ("ui.extraction", extraction, "create_extraction_section"),
        ("ui.harmonization", harmonization, "show_harmonization_section"),
        ("ui.extraction", extraction, "display_structured_clinical_data"),
        ("ui.rag", rag, "create_rag_assistant_section"),
        ("ui.export", export, "_pdf_output_bytes"),
    )
    for module_name, module, function_name in surfaces:
        check(f"{module_name} exposes {function_name}()", hasattr(module, function_name))
    check("app imports with RAG integration", True)
except Exception as error:
    check("app imports with RAG integration", False, repr(error))

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
