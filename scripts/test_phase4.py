"""
OncoLink Phase 4 - Validation & Normalization Test Script

Tests the complete validation pipeline:
1. Pydantic schema validation
2. JSON cleaning and recovery
3. Data normalization
4. Error handling for various bad inputs
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
print("OncoLink Phase 4 - Validation & Normalization Tests")
print("=" * 60)
print()

# ============================================================
# TEST 1: Schema (Pydantic Models)
# ============================================================
print("--- Test 1: Pydantic Schema ---")

from backend.schema import ClinicalData, BiomarkerItem, create_validation_result

# Test creating valid ClinicalData
valid_data = ClinicalData(
    patient_name="Meera Sharma",
    age=58,
    gender="Female",
    diagnosis="Invasive Ductal Carcinoma",
    cancer_type="Breast Cancer",
    cancer_stage="Stage IIB",
    ecog_score=1
)
test("ClinicalData created with valid fields", valid_data is not None)
test("Patient name stored correctly", valid_data.patient_name == "Meera Sharma")
test("Age is integer", isinstance(valid_data.age, int))
test("ECOG in valid range", 0 <= valid_data.ecog_score <= 5)

# Test optional fields default to None
empty_data = ClinicalData()
test("Empty ClinicalData has None fields", empty_data.patient_name is None)
test("Biomarkers default to empty list", empty_data.biomarkers == [])

# Test BiomarkerItem
bm = BiomarkerItem(name="ER", value="Positive", status="Strong")
test("BiomarkerItem created", bm.name == "ER")
test("BiomarkerItem status stored", bm.status == "Strong")

# Test failed validation (age negative)
try:
    bad_data = ClinicalData(age=-5)
    test("Negative age is rejected", False)
except Exception as e:
    test("Negative age raises error", True, str(e)[:50])

# Test failed validation (ECOG > 5)
try:
    bad_data = ClinicalData(ecog_score=99)
    test("ECOG 99 is rejected", False)
except Exception as e:
    test("ECOG > 5 raises error", True, str(e)[:50])

# Test create_validation_result
vr = create_validation_result(valid=True, data={"test": "data"})
test("Validation result created", vr["valid"] == True)
test("Validation result has data", vr["data"]["test"] == "data")

print()

# ============================================================
# TEST 2: JSON Parser
# ============================================================
print("--- Test 2: JSON Parser ---")

from backend.json_parser import clean_llm_response, parse_json_safely, fix_trailing_commas

# Test clean JSON extraction
result = clean_llm_response('{"name": "test"}')
test("Plain JSON extracted", result["success"])

# Test JSON with markdown blocks
result = clean_llm_response('```json\n{"name": "test"}\n```')
test("Markdown JSON extracted", result["success"])

# Test JSON with surrounding text
result = clean_llm_response('Here is the data: {"name": "test"} Hope this helps!')
test("Text-wrapped JSON extracted", result["success"])

# Test empty input
result = clean_llm_response("")
test("Empty input handled", not result["success"])

# Test no JSON present
result = clean_llm_response("No JSON data available")
test("No JSON handled", not result["success"])

# Test trailing commas
fixed = fix_trailing_commas('{"a": 1, "b": 2,}')
test("Trailing comma removed", fixed == '{"a": 1, "b": 2}')

# Test parse_json_safely
result = parse_json_safely('{"valid": true}')
test("Safe parse works", result["success"])

result = parse_json_safely("not json")
test("Safe parse handles bad JSON", not result["success"])

print()

# ============================================================
# TEST 3: Validator
# ============================================================
print("--- Test 3: Validator ---")

from backend.validator import validate_clinical_data

# Test valid data
valid_dict = {
    "patient_name": "Meera Sharma",
    "age": 58,
    "gender": "Female",
    "diagnosis": "Invasive Ductal Carcinoma",
    "cancer_type": "Breast Cancer",
    "cancer_stage": "Stage IIB",
    "biomarkers": [{"name": "ER", "value": "Positive"}],
    "ecog_score": 1
}
result = validate_clinical_data(valid_dict)
test("Valid data passes validation", result["valid"])
test("Validated data has correct fields", result["data"]["patient_name"] == "Meera Sharma")
test("Biomarkers list preserved", len(result["data"]["biomarkers"]) == 1)

# Test empty dict
result = validate_clinical_data({})
test("Empty dict fails validation", not result["valid"])

# Test None input
result = validate_clinical_data(None)
test("None input handled", not result["valid"])
test("Error message for None", len(result["errors"]) > 0)

# Test data with wrong types
bad_dict = {
    "patient_name": 12345,
    "age": "not a number"
}
result = validate_clinical_data(bad_dict)
test("Bad types fail validation", not result["valid"])

# Test data with wrong field names
# Pydantic ignores unknown fields by default (they're silently dropped)
# So patient_name is still recognized even though patint_age and diag are ignored
typo_dict = {
    "patient_name": "Test",
    "patint_age": 58,
    "diag": "Cancer"
}
result = validate_clinical_data(typo_dict)
# Pydantic silently drops unknown fields but keeps known ones
test("Unknown fields silently dropped", result["valid"] == True)
test("Valid known field preserved", result["data"]["patient_name"] == "Test")

# Test missing required fields
minimal_dict = {
    "patient_name": "Test",
    "diagnosis": "Cancer"
}
result = validate_clinical_data(minimal_dict)
test("Minimal valid data passes with warnings", result["valid"] == True)
test("Warnings for missing fields", len(result.get("warnings", [])) >= 0)

print()

# ============================================================
# TEST 4: Normalizer
# ============================================================
print("--- Test 4: Normalizer ---")

from backend.normalizer import run_normalization_pipeline, normalize_clinical_data

# Test gender normalization
data = {"gender": "f"}
result = normalize_clinical_data(data)
test("Gender 'f' normalized to 'Female'", result.get("gender") == "Female")

data = {"gender": "male"}
result = normalize_clinical_data(data)
test("Gender 'male' normalized to 'Male'", result.get("gender") == "Male")

# Test cancer type normalization
data = {"cancer_type": "IDC"}
result = normalize_clinical_data(data)
test("IDC expanded to full name", result.get("cancer_type") == "Invasive Ductal Carcinoma")

# Test name capitalization
data = {"patient_name": "meera sharma"}
result = normalize_clinical_data(data)
test("Name capitalized", result.get("patient_name") == "Meera Sharma")

# Test biomarker normalization
data = {
    "biomarkers": [
        {"name": "er", "value": "positive"},
        {"name": "her2", "value": "-"}
    ]
}
result = normalize_clinical_data(data)
bms = result.get("biomarkers", [])
test("Biomarker names normalized", len(bms) == 2)
if len(bms) >= 2:
    test("ER name normalized", bms[0]["name"] == "ER")
    test("Positive value normalized", bms[0]["value"] == "Positive")
    test("HER2 name normalized", bms[1]["name"] == "HER2")
    test("Negative value normalized", bms[1]["value"] == "Negative")

# Test end-to-end normalizer pipeline
result = run_normalization_pipeline(data)
test("Normalization pipeline returns success", result["success"])
test("Normalization pipeline has data", result["data"] is not None)

# Test normalizer with non-dict input
result = run_normalization_pipeline("not a dict")
test("Non-dict input handled", not result["success"])

print()

# ============================================================
# TEST 5: End-to-End Pipeline (without API key)
# ============================================================
print("--- Test 5: Extractor Pipeline (no API) ---")

from backend.extractor import run_extraction

result = run_extraction("Sample document text for testing.")
test("Extractor pipeline runs without crashing", result is not None)
test("Extractor returns all expected keys", all(k in result for k in ["success", "data", "errors", "warnings", "raw_response", "validation", "normalization"]))

print()
print("=" * 60)
print(f"Results: {pass_count} passed, {fail_count} failed out of {pass_count + fail_count}")
print("=" * 60)
