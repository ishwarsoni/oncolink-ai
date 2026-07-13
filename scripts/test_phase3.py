"""
OncoLink Phase 3 - Test Script
Tests all AI extraction modules without requiring an actual API key.
"""
import sys
import json

sys.path.insert(0, "D:\\AI\\OncoLink")

pass_count = 0
fail_count = 0

def test(name, condition, detail=""):
    global pass_count, fail_count
    if condition:
        pass_count += 1
        print(f"  ✅ {name}")
        if detail:
            print(f"     {detail}")
    else:
        fail_count += 1
        print(f"  ❌ {name}")
        if detail:
            print(f"     {detail}")

print("=" * 60)
print("OncoLink Phase 3 - Module Tests")
print("=" * 60)
print()

# ============================================================
# TEST 1: Module imports
# ============================================================
print("--- Test 1: Module Imports ---")

from backend.llm_client import get_api_key, create_nvidia_client, call_llm
test("llm_client imports OK", True)

from backend.prompt_builder import build_extraction_prompt
test("prompt_builder imports OK", True)

from backend.models import CLINICAL_SCHEMA, get_field_label, get_required_fields
test("models imports OK", True)

from backend.extractor import run_extraction, attempt_json_recovery, extract_from_multiple_documents
test("extractor imports OK", True)

print()

# ============================================================
# TEST 2: Prompt Builder
# ============================================================
print("--- Test 2: Prompt Builder ---")

system_prompt, user_prompt = build_extraction_prompt("Sample medical text.")
test("System prompt is not empty", len(system_prompt) > 0, f"Length: {len(system_prompt)} chars")
test("User prompt is not empty", len(user_prompt) > 0, f"Length: {len(user_prompt)} chars")
test("User prompt contains document text", "Sample medical text." in user_prompt)
test("System prompt mentions JSON", "JSON" in system_prompt)
test("System prompt says no markdown", "markdown" in system_prompt)

print()

# ============================================================
# TEST 3: Models Schema
# ============================================================
print("--- Test 3: Models Schema ---")

test("Schema has correct number of fields", len(CLINICAL_SCHEMA) >= 10)
test("ECOG field exists", "ecog_score" in CLINICAL_SCHEMA)
test("Diagnosis field exists", "diagnosis" in CLINICAL_SCHEMA)

label = get_field_label("ecog_score")
test("get_field_label works", label == "ECOG Score", f"Got: {label}")

required = get_required_fields()
test("Required fields are defined", len(required) > 0, f"Required: {required}")
test("patient_name is required", "patient_name" in required)
test("diagnosis is required", "diagnosis" in required)

print()

# ============================================================
# TEST 4: JSON Recovery
# ============================================================
print("--- Test 4: JSON Recovery ---")

# Test: Plain JSON
recovered = attempt_json_recovery('{"patient_name": "Test", "age": 45}')
test("Plain JSON extraction", recovered == '{"patient_name": "Test", "age": 45}')

# Test: JSON with surrounding text
recovered = attempt_json_recovery('Here is the data: {"name": "test"} Thanks!')
test("JSON with surrounding text", recovered is not None, f"Got: {recovered}")

# Test: Markdown code blocks
recovered = attempt_json_recovery('```json\n{"patient_name": "Meera"}\n```')
test("Markdown code block JSON", recovered is not None, f"Got: {recovered}")

# Test: Cleaned JSON parses correctly
parsed = json.loads(recovered)
test("Recovered JSON is valid", parsed["patient_name"] == "Meera")

# Test: No JSON present
recovered = attempt_json_recovery("The patient has no clinical data available.")
test("No JSON returns None", recovered is None)

print()

# ============================================================
# TEST 5: Extractor handles missing API key
# ============================================================
print("--- Test 5: Missing API Key Handling ---")

result = run_extraction("test document text")
test("Extractor returns result dict", isinstance(result, dict))
test("Extractor handles missing key", result["success"] == False)
if not result["success"]:
    test("Error message present", len(result["error"]) > 0)
    test("Error mentions API key", "API" in result["error"] or "key" in result["error"].lower())

print()

# ============================================================
# TEST 6: extract_from_multiple_documents
# ============================================================
print("--- Test 6: Multi-document extraction ---")

# Create mock processed files (simulating what document_loader returns)
mock_files = [
    {
        "filename": "referral.txt",
        "text": "Patient is a 58-year-old female with left breast mass.",
        "file_type": "TXT",
        "success": True
    },
    {
        "filename": "pathology.pdf",
        "text": "Diagnosis: Invasive ductal carcinoma, ER+/PR+/HER2-",
        "file_type": "PDF",
        "success": True
    }
]

result = extract_from_multiple_documents(mock_files)
test("Multi-doc returns result dict", isinstance(result, dict))
# Should fail because no API key, but should not crash

# Test with empty files list
empty_result = extract_from_multiple_documents([])
test("Empty files handled", empty_result["success"] == False)

# Test with failed files only
failed_files = [
    {
        "filename": "broken.pdf",
        "text": "Error reading file",
        "file_type": "PDF",
        "success": False
    }
]
failed_result = extract_from_multiple_documents(failed_files)
test("All-failed files handled", failed_result["success"] == False)

print()
print("=" * 60)
print(f"Results: {pass_count} passed, {fail_count} failed out of {pass_count + fail_count}")
print("=" * 60)
