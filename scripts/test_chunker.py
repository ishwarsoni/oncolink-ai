"""Tests for backend/chunker.py (no model downloads required)."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend.chunker import (
    chunk_document,
    chunk_documents,
    count_tokens,
    token_count_method,
)
from rag_test_helpers import synthetic_document

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
print("Chunker tests")
print("=" * 70)
print(f"Token counting: {token_count_method()}")

long_text = """
PATIENT INFORMATION:
Name: Test Patient
Age: 58 years
Gender: Female

DIAGNOSIS:
Invasive ductal carcinoma of the left breast, stage IIB.

BIOMARKERS:
PD-L1: 45% positive.
EGFR: Negative.
ALK: Negative.

MEDICATIONS:
Pembrolizumab 200 mg every three weeks.
Ondansetron as needed for nausea.

FOLLOW-UP PLAN:
Repeat imaging in three months and monitor for immune-related adverse events.
""".strip()

document = synthetic_document(
    patient_id="case_test",
    document_id="doc_test",
    filename="report.txt",
    text=long_text,
    page_number=1,
    document_date="2026-01-15",
)

chunks = chunk_document(document, chunk_size=40, chunk_overlap=10)
check("Produces multiple chunks", len(chunks) > 1, f"got {len(chunks)}")
check("Every chunk has a stable chunk_id", all(c["chunk_id"] for c in chunks))
check("Chunk IDs are unique", len({c["chunk_id"] for c in chunks}) == len(chunks))
check("Patient ID preserved", all(c["patient_id"] == "case_test" for c in chunks))
check("Document ID preserved", all(c["document_id"] == "doc_test" for c in chunks))
check("Filename preserved", all(c["filename"] == "report.txt" for c in chunks))
check("Page number preserved", all(c["page_number"] == 1 for c in chunks))
check("Document date preserved", all(c["document_date"] == "2026-01-15" for c in chunks))
check("Content hash preserved", all(c["content_hash"] for c in chunks))
check("Token counts present", all(c["token_count"] > 0 for c in chunks))
check(
    "Chunk token counts respect the budget (with boundary slack)",
    all(c["token_count"] <= 40 + 20 for c in chunks),
    f"max={max(c['token_count'] for c in chunks)}",
)

headings = {c["section_heading"] for c in chunks}
check("Section headings are preserved", "DIAGNOSIS" in headings or "BIOMARKERS" in headings, headings)

# Overlap is tested inside a long section.  Section headings intentionally
# start a fresh chunk, so overlap is not forced across unrelated sections.
overlap_text = (
    "The patient was diagnosed with invasive ductal carcinoma of the left breast. "
    "The tumor measured three centimeters in the upper outer quadrant. "
    "Imaging showed no distant metastases. "
    "The patient completed six cycles of chemotherapy. "
    "Follow-up imaging is scheduled in three months. "
    "The patient tolerates the current medication well. "
    "No new adverse events were reported this cycle. "
    "Laboratory values remained within normal limits. "
    "The care team will review the next scan at the follow-up visit."
)
overlap_document = synthetic_document(
    patient_id="case_test",
    document_id="doc_overlap",
    filename="long_note.txt",
    text=overlap_text,
)
overlap_chunks = chunk_document(overlap_document, chunk_size=40, chunk_overlap=12)

shared = False
for first, second in zip(overlap_chunks, overlap_chunks[1:]):
    first_words = set(first["text"].split())
    second_words = set(second["text"].split())
    if len(first_words & second_words) >= 3:
        shared = True
        break
check("Adjacent chunks overlap inside a long section", shared)

# Overlap = 0 should produce fewer or equal chunks than a large overlap.
no_overlap = chunk_document(overlap_document, chunk_size=40, chunk_overlap=0)
large_overlap = chunk_document(overlap_document, chunk_size=40, chunk_overlap=25)
check(
    "Overlap is configurable",
    len(large_overlap) >= len(no_overlap),
    f"no_overlap={len(no_overlap)} large_overlap={len(large_overlap)}",
)

# Multi-page metadata.
multi_page = synthetic_document(
    patient_id="case_test",
    document_id="doc_pages",
    filename="scan.pdf",
    text="Page one text about EGFR.\nPage two text about ALK.",
)
multi_page["pages"] = [
    {"page_number": 1, "text": "IMAGING:\nCT chest shows a right upper lobe mass. EGFR testing requested."},
    {"page_number": 2, "text": "MOLECULAR:\nEGFR exon 19 deletion detected. ALK negative."},
]
page_chunks = chunk_document(multi_page, chunk_size=30, chunk_overlap=5)
check("Multi-page chunks keep page 1", any(c["page_number"] == 1 for c in page_chunks))
check("Multi-page chunks keep page 2", any(c["page_number"] == 2 for c in page_chunks))

# Duplicate content should not produce duplicate chunks.
duplicate_doc = synthetic_document(
    patient_id="case_test",
    document_id="doc_dup",
    filename="duplicate.txt",
    text="Same text repeated.",
)
duplicate_doc["pages"] = [
    {"page_number": 1, "text": "DIAGNOSIS:\nNSCLC stage IIIA."},
    {"page_number": 2, "text": "DIAGNOSIS:\nNSCLC stage IIIA."},
]
duplicate_chunks = chunk_document(duplicate_doc, chunk_size=40, chunk_overlap=0)
check("Duplicate text is deduplicated", len(duplicate_chunks) == 1, len(duplicate_chunks))

# Empty and malformed inputs.
check("Empty document returns no chunks", chunk_document({"text": ""}) == [])
check("Non-dictionary input returns no chunks", chunk_document("not a document") == [])
check("Empty document list returns no chunks", chunk_documents([]) == [])

# Token counting is not character counting.
sample = "The patient has EGFR exon 19 deletion."
check("Token count is smaller than character count", count_tokens(sample) < len(sample))
check("Token count is positive", count_tokens(sample) > 0)

print()
print(f"Results: {PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
