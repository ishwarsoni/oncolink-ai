"""
=============================================================================
schema.py - Pydantic Models for Clinical Data Validation
=============================================================================

What this module does:
    Defines Pydantic data models that represent the structure of our
    clinical data. These models enforce:
    1. What fields are expected (no unexpected fields)
    2. What data types each field should be (string, int, list, etc.)
    3. Which fields are optional vs required
    4. Validation rules (age must be positive, ECOG must be 0-5, etc.)

What is Pydantic?
    Pydantic is a Python library that validates data at runtime.
    You define a "model" (a class that inherits from BaseModel) and
    specify each field's type. When you create an instance, Pydantic
    automatically checks that all values match their types.
    
    Example:
        class Person(BaseModel):
            name: str
            age: int
        
        person = Person(name="Meera", age=58)  # Works!
        person = Person(name="Meera", age="old")  # Fails validation!

Why use Pydantic for LLM outputs?
    LLMs are probabilistic — they might return anything. Pydantic acts
    as a "safety net" that catches:
    - Wrong field names (typos like "patint_name")
    - Wrong types (age as a string like "fifty eight")
    - Missing required fields
    - Extra fields we didn't ask for

Key Pydantic concepts:
    BaseModel: The parent class all models inherit from
    Optional[type]: A field that can be None (e.g., Optional[str])
    List[type]: A field that is a list of items
    Field(): Adds extra validation like default values, descriptions
    model_dump(): Converts the model back to a dictionary

=============================================================================
"""

# Import Pydantic's BaseModel and field types
# BaseModel is the foundation — all our models inherit from it
from pydantic import BaseModel, Field

# Import typing helpers for flexible type definitions
# Optional means the field can be None (not found in document)
# List means the field is a list of items
from typing import Optional, List


# =============================================================================
# BiomarkerItem Model
# =============================================================================
# This model represents a single biomarker (e.g., ER, PR, HER2, Ki-67).
# Biomarkers are stored as a list of these items inside ClinicalData.
#
# Why a separate model:
#   Biomarkers have a complex structure (name + value + status).
#   A list of dictionaries is hard to validate.
#   A Pydantic model makes each biomarker's structure explicit.
# =============================================================================

class BiomarkerItem(BaseModel):
    """
    Represents a single biomarker found in the patient's reports.
    
    Fields:
        name (str): The biomarker name (e.g., "ER", "PR", "HER2", "Ki-67")
        value (str): The measured value (e.g., "Positive", "90%", "25%")
        status (Optional[str]): Additional status info (e.g., "Strong intensity")
    
    All fields have default values so partial data still works.
    """
    name: str = ""
    value: str = ""
    status: Optional[str] = None


# =============================================================================
# ClinicalData Model
# =============================================================================
# This is the MAIN model — it represents ALL clinical information for a patient.
# Every field the LLM extracts is validated against this model.
#
# Optional fields:
#   Most fields are Optional because the LLM might not find them in documents.
#   If a field is not found, it defaults to None (null in JSON).
#
# Field validation:
#   The Field() function adds extra rules like ge=0 (greater than or equal to 0).
#   This catches invalid values like negative age or ECOG score of 99.
# =============================================================================

class ClinicalData(BaseModel):
    """
    Structured clinical data extracted from oncology documents.
    
    Each field maps to a clinical concept. Optional fields use None
    to indicate "not found in documents" rather than "empty string".
    
    Python concepts used:
        - Class inheritance (ClinicalData(BaseModel))
        - Type annotations (name: str, age: Optional[int])
        - Pydantic's Field() for validation rules
        - Default values for graceful handling of missing data
    """
    
    # ---- Patient Identity ----
    # Optional[str] means: this field can be None (missing) or a string
    # If the LLM doesn't find the patient name, it's None, not ""
    patient_name: Optional[str] = None
    
    # Optional[int] means: this field can be None or an integer
    # ge=0 means "greater than or equal to 0" — catches negative ages
    age: Optional[int] = Field(default=None, ge=0, le=150)
    
    # Simple optional string for gender
    gender: Optional[str] = None
    
    # ---- Cancer Information ----
    # These are the core clinical fields
    diagnosis: Optional[str] = None
    cancer_type: Optional[str] = None
    
    # Cancer stage including TNM classification
    cancer_stage: Optional[str] = None
    
    # ---- Biomarkers ----
    # List[BiomarkerItem] means this is a list of BiomarkerItem objects
    # Default is an empty list (no biomarkers found)
    biomarkers: List[BiomarkerItem] = []
    
    # ---- Treatment ----
    current_medication: Optional[str] = None
    previous_treatment: Optional[str] = None
    
    # ---- Performance Status ----
    # ECOG score must be 0-5 (standard clinical scale)
    # ge=0 means >= 0, le=5 means <= 5
    ecog_score: Optional[int] = Field(default=None, ge=0, le=5)
    
    # ---- Plan ----
    follow_up_plan: Optional[str] = None
    next_steps: Optional[str] = None

    # ---- Adverse Events ----
    adverse_events: Optional[str] = None


# =============================================================================
# ValidationResult Model
# =============================================================================
# This model represents the RESULT of validation.
# It contains:
#   - valid: Whether all validation checks passed
#   - data: The validated (and possibly corrected) ClinicalData
#   - errors: List of validation error messages
#   - warnings: List of validation warnings (non-critical issues)
#
# This is NOT validated by Pydantic itself — it's a container for results.
# We use a regular dataclass/dict pattern instead.
# =============================================================================

# We don't create a Pydantic model for this.
# Instead, we use a simple function that returns a dictionary.
# This keeps the code simpler for a beginner.


def create_validation_result(valid, data, errors=None, warnings=None):
    """
    Create a standardized validation result dictionary.
    
    Parameters:
        valid (bool): True if data passed all validation checks
        data (dict or None): The validated clinical data
        errors (list): List of error messages (critical issues)
        warnings (list): List of warning messages (non-critical)
    
    Returns:
        dict: A standardized result dictionary
    
    This is a helper function that ensures all validation results
    have the same structure. Consistency makes the calling code simpler.
    """
    if errors is None:
        errors = []
    if warnings is None:
        warnings = []
    
    return {
        "valid": valid,
        "data": data,
        "errors": errors,
        "warnings": warnings
    }
