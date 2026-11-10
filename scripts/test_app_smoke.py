"""Streamlit smoke tests for the OncoLink workspace shell.

The AppTest runner executes the real ``app.py``.  It verifies the three
workspace views, the structured-record renderer, chat rendering and the
existing export helpers.  It does not call NVIDIA or download a model.
"""

import html
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from ui.export import _pdf_output_bytes, _text_to_pdf_bytes
from ui.styles import _CSS

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


def markdown_text(app):
    return html.unescape("\n".join(item.value for item in app.markdown))


def button_labels(app):
    labels = [button.label for button in app.button]
    # Download buttons are not exposed as typed AppTest elements in this
    # Streamlit version, but their elements are still discoverable.
    labels.extend(element.label for element in app.get("download_button"))
    return labels


print("=" * 70)
print("Streamlit application smoke tests")
print("=" * 70)

# ---------------------------------------------------------------------------
# Initial page load
# ---------------------------------------------------------------------------
print("\n--- Initial render ---")

app = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
app.run()

check("Application runs without exceptions", not app.exception, app.exception)
check("Page title is rendered", any("OncoLink" in item.value for item in app.markdown))
check("A patient case ID exists", bool(app.session_state["patient_id"]))
check(
    "Case ID is non-identifying",
    str(app.session_state["patient_id"]).startswith("case_"),
    app.session_state["patient_id"],
)
check("Chat starts empty", app.session_state["rag_chat"] == [])
check(
    "First-run intake is rendered",
    "Nothing processed yet" in markdown_text(app),
)

# ---------------------------------------------------------------------------
# Workspace views with processed documents
# ---------------------------------------------------------------------------
print("\n--- Workspace views ---")

app.session_state["processed_files"] = [
    {
        "filename": "sample.txt",
        "text": "Sample clinical document. Diagnosis: breast cancer.",
        "file_type": "TXT",
        "success": True,
        "pages": [
            {
                "page_number": None,
                "text": "Sample clinical document. Diagnosis: breast cancer.",
            }
        ],
    },
    {
        "filename": "pathology.txt",
        "text": "Pathology report. Stage IIA breast cancer.",
        "file_type": "TXT",
        "success": True,
        "pages": [
            {
                "page_number": None,
                "text": "Pathology report. Stage IIA breast cancer.",
            }
        ],
    },
]
app.session_state["processing_done"] = True

# This is the state the deployed build crashed on: documents have been
# processed, and ingestion explicitly cleared any previous extraction results.
for key in (
    "extraction_result",
    "extraction_done",
    "harmonized_data",
    "conflicts",
    "summary",
):
    app.session_state[key] = None
app.session_state["harmonization_done"] = False
app.run()

check("Page still runs after upload state", not app.exception, app.exception)
check(
    "Processed case with no extraction result renders",
    not app.exception,
    app.exception,
)
check(
    "Extraction call-to-action is available",
    "Run AI extraction" in button_labels(app),
    button_labels(app),
)
check(
    "Patient Record view is the default",
    app.session_state["workspace_view"] == "Patient Record",
)
check(
    "Extraction section is rendered",
    "AI Clinical Extraction" in markdown_text(app),
)
check(
    "Patient Record heading is rendered",
    "Patient Record" in markdown_text(app),
)
check(
    "Navigation offers all three views",
    len(app.radio) == 1
    and list(app.radio[0].options)
    == ["Patient Record", "AI Clinical Chat", "Uploaded Documents"],
    [radio.options for radio in app.radio],
)

# --- Chat view -------------------------------------------------------------
app.session_state["workspace_view"] = "AI Clinical Chat"
app.run()

check("Chat view runs without exceptions", not app.exception, app.exception)
check(
    "RAG assistant is rendered",
    "Clinical RAG Assistant" in markdown_text(app),
)
check("Chat input is present in the chat view", len(app.chat_input) == 1)
labels = button_labels(app)
check(
    "Suggested questions are rendered",
    "What is the patient's primary diagnosis?" in labels,
    labels,
)
check(
    "Case state survives the view switch",
    len(app.session_state["processed_files"]) == 2,
)
check("Chat is still empty", app.session_state["rag_chat"] == [])

# --- Documents view --------------------------------------------------------
app.session_state["workspace_view"] = "Uploaded Documents"
app.run()

check("Documents view runs without exceptions", not app.exception, app.exception)
check(
    "Extracted-text view is rendered",
    "Extracted Text" in markdown_text(app),
)
check(
    "Both processed documents are listed",
    "sample.txt" in markdown_text(app) and "pathology.txt" in markdown_text(app),
)
check("Chat input is absent outside the chat view", len(app.chat_input) == 0)

# ---------------------------------------------------------------------------
# Structured record rendering
# ---------------------------------------------------------------------------
print("\n--- Structured record ---")

clinical_data = {
    "patient_name": "Test Patient",
    "age": 54,
    "gender": "Female",
    "diagnosis": "Invasive ductal carcinoma of the left breast",
    "cancer_type": "Breast cancer",
    "cancer_stage": "Stage IIA",
    "biomarkers": [
        {"name": "ER", "value": "95%", "status": "positive"},
        {"name": "HER2", "value": None, "status": None},
    ],
    "current_medication": "Tamoxifen",
    "previous_treatment": "Lumpectomy",
    "ecog_score": 0,
    "follow_up_plan": "Oncology review every three months",
    "next_steps": "Repeat imaging in six months",
    "adverse_events": None,
}

app.session_state["extraction_result"] = {
    "success": True,
    "data": clinical_data,
    "warnings": ["Age differed between documents."],
    "errors": [],
}
app.session_state["extraction_done"] = True
app.session_state["harmonized_data"] = {
    "success": True,
    "data": clinical_data,
    "documents_merged": 2,
    "field_sources": {
        "patient_name": "sample.txt",
        "diagnosis": "pathology.txt",
        "cancer_stage": "pathology.txt",
    },
    "warnings": ["Age differed between documents."],
}
app.session_state["conflicts"] = {
    "success": True,
    "conflicts": [
        {
            "severity": "high",
            "field": "age",
            "description": "Age differs between documents",
            "sources": [["sample.txt", "54"], ["pathology.txt", "56"]],
        }
    ],
}
app.session_state["summary"] = {"success": True, "summary": "Test patient summary."}
app.session_state["harmonization_done"] = True
app.session_state["workspace_view"] = "Patient Record"
app.run()

record_text = markdown_text(app)
check("Record view runs without exceptions", not app.exception, app.exception)
for section in (
    "Patient Information",
    "Diagnosis & Cancer Stage",
    "Biomarkers & Test Results",
    "Current Medication",
    "Previous Treatment",
    "ECOG Performance Status",
    "Follow-up Plan & Next Steps",
    "Adverse Events",
    "Document Sources & Provenance",
    "Conflicts & Warnings",
):
    check(f"Record section rendered: {section}", section in record_text)

check("Missing values are explicit", "Not documented" in record_text)
check(
    "Provenance names the source document",
    "pathology.txt" in record_text,
)
check(
    "Conflicts are visible with both values",
    "Age differs between documents" in record_text and "56" in record_text,
)
check(
    "Conflicting fields are flagged inline",
    "Conflict reported: Age" in record_text,
)
check(
    "Record workflow offers a distinct re-extraction control",
    "Re-run AI extraction" in button_labels(app)
    and "Run AI extraction" not in button_labels(app),
    button_labels(app),
)
check(
    "Harmonization source is labelled",
    "Harmonized from 2 document(s)" in record_text,
)
check(
    "Summary and exports remain available",
    "Test patient summary." in record_text
    and "Download JSON" in button_labels(app),
)

# ---------------------------------------------------------------------------
# Chat history rendering and clearing
# ---------------------------------------------------------------------------
print("\n--- Chat history ---")

app.session_state["rag_chat"] = [
    {"role": "user", "content": "What is the diagnosis?"},
    {
        "role": "assistant",
        "content": "Invasive ductal carcinoma.",
        "result": {
            "success": True,
            "answer": "Invasive ductal carcinoma.",
            "sources": [
                {
                    "filename": "pathology.txt",
                    "page_label": "Page not available",
                    "quote": "Invasive ductal carcinoma",
                }
            ],
            "retrieved_evidence": [],
            "unresolved_contradictions": ["Stage differs between notes."],
            "insufficient_evidence": False,
            "warnings": [],
        },
    },
]
app.session_state["workspace_view"] = "AI Clinical Chat"
app.run()

chat_text = markdown_text(app)
check("Chat history runs without exceptions", not app.exception, app.exception)
check("Stored answer is rendered", "Invasive ductal carcinoma." in chat_text)
check("Filename citation is rendered", "pathology.txt" in chat_text)
check("Contradiction indicator is rendered", "Stage differs between notes." in chat_text)

clear_buttons = [button for button in app.button if button.label == "Clear chat"]
if clear_buttons:
    clear_buttons[0].click()
    app.run()
check("Clear chat removes history", app.session_state["rag_chat"] == [])
check("Clear chat does not remove documents", len(app.session_state["processed_files"]) == 2)

# ---------------------------------------------------------------------------
# New case reset
# ---------------------------------------------------------------------------
print("\n--- Case reset ---")

original_case = app.session_state["patient_id"]
new_case_buttons = [button for button in app.button if button.label == "Start new case"]
if new_case_buttons:
    new_case_buttons[0].click()
    app.run()

check("New-case reset runs without exceptions", not app.exception, app.exception)
check("New-case reset changes the case ID", app.session_state["patient_id"] != original_case)
check(
    "New-case reset clears documents",
    "processed_files" not in app.session_state,
)
check("New-case reset clears chat", app.session_state["rag_chat"] == [])
check(
    "New-case reset returns to the record view",
    app.session_state["workspace_view"] == "Patient Record",
)

# ---------------------------------------------------------------------------
# Theme font scope
# ---------------------------------------------------------------------------
print("\n--- Theme fonts ---")

check(
    "App font override excludes Material icon spans",
    '[class*="st-"]:not([data-testid="stIconMaterial"])' in _CSS,
    "Streamlit renders the expander chevron as a Material Symbols ligature; "
    "overriding its font makes the raw icon name overlap the label.",
)

# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------
print("\n--- Exports ---")

pdf_bytes = _text_to_pdf_bytes("Patient summary\nDiagnosis: breast cancer")
check("Text-to-PDF export returns bytes", isinstance(pdf_bytes, bytes) and pdf_bytes.startswith(b"%PDF"))


class _FakePdf:
    def output(self):
        return bytearray(b"%PDF-1.4 test")


check("PDF output helper normalises bytes", _pdf_output_bytes(_FakePdf()) == b"%PDF-1.4 test")
check("PDF bytes are non-empty", len(pdf_bytes) > 100)

print()
print("=" * 70)
print(f"Results: {PASS} passed, {FAIL} failed")
print("=" * 70)
sys.exit(1 if FAIL else 0)
