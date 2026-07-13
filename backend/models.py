"""
=============================================================================
models.py - Clinical Data Schema Definition
=============================================================================

What this module does:
    Defines the expected structure of our clinical data.
    This is a REFERENCE that documents what fields the LLM should extract
    and what data types they should have.

Why we need this:
    When the LLM returns JSON, we need to know:
    1. What fields to expect (so we don't miss any)
    2. What data types each field should be (string, number, list, etc.)
    3. What the display labels should be (user-friendly names)
    4. Which fields are critical vs optional

    This module serves as the "source of truth" for our data structure.
    It will be used by:
    - The UI to know which fields to display
    - Future validation modules to check data quality
    - Future export modules to format the output

Why NOT Pydantic yet:
    Pydantic is a Python library that enforces data types at runtime.
    It's powerful but adds complexity. For Phase 3:
    - We're learning LLM APIs first — don't overload with frameworks
    - A Python dictionary is simpler to understand
    - We'll introduce Pydantic in Phase 4 for validation

=============================================================================
"""


# =============================================================================
# Clinical Data Schema
# =============================================================================
# This dictionary defines every field we want the LLM to extract.
# Each field has:
#   - "type": The expected data type (for documentation/validation)
#   - "label": A user-friendly name for the UI
#   - "description": What this field means in clinical context
#   - "required": Whether this field is critical for patient care
#
# This is NOT used for validation (no Pydantic yet).
# It serves as documentation and reference for the UI.
# =============================================================================

CLINICAL_SCHEMA = {
    "patient_name": {
        "type": "string",
        "label": "Patient Name",
        "description": "Full name of the patient",
        "required": True
    },
    "age": {
        "type": "number",
        "label": "Age",
        "description": "Patient's age in years",
        "required": True
    },
    "gender": {
        "type": "string",
        "label": "Gender",
        "description": "Patient's gender (Male/Female/Other)",
        "required": True
    },
    "diagnosis": {
        "type": "string",
        "label": "Diagnosis",
        "description": "Complete diagnosis description",
        "required": True
    },
    "cancer_type": {
        "type": "string",
        "label": "Cancer Type",
        "description": "Specific type of cancer",
        "required": True
    },
    "cancer_stage": {
        "type": "string",
        "label": "Cancer Stage",
        "description": "Stage including TNM classification if available",
        "required": True
    },
    "biomarkers": {
        "type": "list",
        "label": "Biomarkers",
        "description": "List of biomarkers with name, value, and status",
        "required": False
    },
    "current_medication": {
        "type": "string",
        "label": "Current Medication",
        "description": "Current medications the patient is taking",
        "required": False
    },
    "previous_treatment": {
        "type": "string",
        "label": "Previous Treatment",
        "description": "Treatments the patient has already received",
        "required": False
    },
    "ecog_score": {
        "type": "number",
        "label": "ECOG Score",
        "description": "ECOG performance status (0-5) or null if not found",
        "required": False
    },
    "follow_up_plan": {
        "type": "string",
        "label": "Follow-up Plan",
        "description": "Recommended follow-up schedule",
        "required": False
    },
    "next_steps": {
        "type": "string",
        "label": "Next Steps",
        "description": "Recommended next actions or treatment plan",
        "required": False
    },
    "adverse_events": {
        "type": "string",
        "label": "Adverse Events",
        "description": "Side effects, toxicities, or adverse reactions to treatment",
        "required": False
    }
}


# =============================================================================
# Helper function: Get field labels for the UI
# =============================================================================

def get_field_label(field_name):
    """
    Get the user-friendly label for a clinical field.
    
    Parameters:
        field_name (str): The internal field name (e.g., "ecog_score")
    
    Returns:
        str: The display label (e.g., "ECOG Score")
             If the field is not found in schema, returns the field name
             with underscores replaced by spaces and title-cased.
    
    Python concepts used:
        - Dictionary .get() with a default value
        - .title() method: converts "ecog_score" to "Ecog_Score"
        - .replace() method: converts "Ecog_Score" to "Ecog Score"
    """
    
    # Check if this field exists in our schema
    if field_name in CLINICAL_SCHEMA:
        # Return the user-friendly label
        return CLINICAL_SCHEMA[field_name]["label"]
    else:
        # Fallback: convert field_name to a readable label
        # Example: "ecog_score" -> "ecog score" -> "Ecog Score"
        readable_name = field_name.replace("_", " ")
        return readable_name.title()


def get_required_fields():
    """
    Get a list of field names that are marked as required.
    
    Returns:
        list: Names of all required fields
    
    Python concepts used:
        - List comprehension: [item for item in collection if condition]
        - Iterating over dictionary items with .items()
    """
    
    required_fields = []
    
    for field_name, field_info in CLINICAL_SCHEMA.items():
        if field_info["required"]:
            required_fields.append(field_name)
    
    return required_fields
