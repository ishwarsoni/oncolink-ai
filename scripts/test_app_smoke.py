"""Streamlit smoke tests for the refactored application shell.

The AppTest runner executes the real ``app.py``.  It verifies that the page
renders without exceptions after the layer refactor and that the export
helpers still work.  It does not call NVIDIA or download a model.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from streamlit.testing.v1 import AppTest

from ui.export import _pdf_output_bytes, _text_to_pdf_bytes

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

# ---------------------------------------------------------------------------
# Rendered sections after documents exist
# ---------------------------------------------------------------------------
print("\n--- Sections with processed documents ---")

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
    }
]
app.session_state["processing_done"] = True
app.run()

check("Page still runs after upload state", not app.exception, app.exception)
check(
    "Extraction section is rendered",
    any("AI Clinical Extraction" in item.value for item in app.markdown),
)
check(
    "RAG assistant is rendered",
    any("Clinical RAG Assistant" in item.value for item in app.markdown),
)
check(
    "Extracted-text view is rendered",
    any("Extracted Text" in item.value for item in app.markdown),
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
