"""Test Phase 5: Harmonization, conflict detection, and summary"""
import sys, os, json

sys.path.insert(0, "D:\\AI\\OncoLink")
os.chdir("D:\\AI\\OncoLink")

from dotenv import load_dotenv
load_dotenv()

from backend.document_loader import process_uploaded_file
from backend.extractor import extract_each_document
from backend.harmonizer import harmonize_extractions
from backend.conflict_detector import detect_conflicts
from backend.summarizer import generate_patient_summary


class MockFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._bytes = f.read()
    def read(self):
        return self._bytes


files = [
    MockFile("data/test_patient.txt"),
    MockFile("data/referral.txt"),
]

processed = []
for f in files:
    result = process_uploaded_file(f)
    processed.append(result)
    print(f"Read: {result['filename']} ({len(result['text'])} chars)")

print()
print("Running per-document extraction (this calls the LLM)...")
per_doc = extract_each_document(processed)
for r in per_doc:
    print(f"  {r['filename']}: success={r['success']}")

print()
print("Harmonizing...")
harmonized = harmonize_extractions(per_doc)
print(f"  Success: {harmonized['success']}")
print(f"  Documents merged: {harmonized['documents_merged']}")

print()
print("Checking conflicts...")
conflicts = detect_conflicts(per_doc)
print(f"  Conflicts found: {len(conflicts['conflicts'])}")
for c in conflicts["conflicts"]:
    print(f"  [{c['severity']}] {c['description']}")

print()
print("Generating summary...")
summary = generate_patient_summary(harmonized["data"], harmonized.get("field_sources"))
print(f"  Success: {summary['success']}")
if summary["success"]:
    print()
    print(summary["summary"])
