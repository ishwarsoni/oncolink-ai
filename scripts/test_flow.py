"""Quick end-to-end test with API key"""
import os, sys, json
sys.path.insert(0, "D:\\AI\\OncoLink")
os.chdir("D:\\AI\\OncoLink")

from dotenv import load_dotenv
load_dotenv()

key = os.getenv("NVIDIA_API_KEY")
if not key or key == "nvapi-your-key-here":
    print("API key not configured. Set it in .env file.")
    sys.exit(1)

print(f"API Key loaded: {key[:15]}...")

from backend.document_loader import process_uploaded_file

class MockFile:
    def __init__(self, path):
        self.name = os.path.basename(path)
        with open(path, "rb") as f:
            self._bytes = f.read()
    def read(self):
        return self._bytes

mock = MockFile("data/test_patient.txt")
read_result = process_uploaded_file(mock)
print(f"Document read: {read_result['success']}")
print(f"Text length: {len(read_result['text'])} chars")

from backend.extractor import run_extraction
extract_result = run_extraction(read_result["text"])
print(f"\nExtraction success: {extract_result['success']}")

if extract_result["success"]:
    print("\n=== EXTRACTED DATA ===")
    print(json.dumps(extract_result["data"], indent=2))
    print(f"\nWarnings: {extract_result['warnings']}")
else:
    print(f"Errors: {extract_result['errors']}")
