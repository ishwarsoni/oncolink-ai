"""
Test script for Phase 2 document ingestion.
Tests all three file types: TXT, DOCX, PDF.
"""
import sys
import os

# Add the project root to Python's import path
sys.path.insert(0, "D:\\AI\\OncoLink")

from backend.document_loader import process_uploaded_file


class MockFile:
    """
    A mock version of Streamlit's UploadedFile.
    
    Streamlit's UploadedFile has .name and .read().
    We mimic that here so we can test without the web UI.
    """
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._bytes = f.read()
    
    def read(self):
        return self._bytes


def test_file(file_path, expected_type):
    """
    Test reading a single file and print results.
    """
    mock_file = MockFile(file_path)
    result = process_uploaded_file(mock_file)
    
    success = result["success"]
    text = result["text"]
    file_type = result["file_type"]
    
    # Show first 100 characters of extracted text
    preview = text[:100].strip().replace("\n", " | ")
    
    status = "PASS" if success else "FAIL"
    print(f"[{status}] {file_type}: {mock_file.name}")
    print(f"       Type detected: {file_type}")
    print(f"       Text length: {len(text)} characters")
    print(f"       Preview: {preview}")
    print()


# Test all three file types
print("=" * 60)
print("OncoLink Phase 2 - Document Ingestion Test")
print("=" * 60)
print()

test_file("D:\\AI\\OncoLink\\data\\referral.txt", "TXT")
test_file("D:\\AI\\OncoLink\\data\\lab_report.docx", "DOCX")
test_file("D:\\AI\\OncoLink\\data\\pathology_report.pdf", "PDF")

print("=" * 60)
print("All tests complete!")
