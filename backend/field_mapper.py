"""
=============================================================================
field_mapper.py - Standardize Field Names Across Documents
=============================================================================

What this module does:
    Maps different names for the same clinical concept to a single canonical
    (standard) field name. This is essential when merging data from multiple
    sources that might use different terminology.

Why different names exist:
    Different hospitals and labs use different naming conventions:
    - Hospital A: "Cancer Stage"
    - Hospital B: "Tumor Stage"  
    - Hospital C: "Stage"
    
    All three mean the same thing, but a computer sees them as different.
    The field mapper converts all variations to one canonical name.

What is Data Mapping?
    Data mapping is the process of creating relationships between data
    fields from different sources. It answers: "What does this field
    in Document A correspond to in Document B?"

    In our case, since all LLM extractions use the same Pydantic schema,
    field names should already be consistent. This module handles edge
    cases and future-proofs the system for external data sources.

=============================================================================
"""

# Import logging
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# Canonical Field Names
# =============================================================================
# This dictionary defines the "source of truth" field names.
# Every field from any document should be mapped to one of these.
# We use snake_case for Python compatibility.
#
# The key is the canonical name, the value is a list of synonyms
# that might appear in different sources.
# =============================================================================

# Synonyms for each canonical field
# The first synonym in each list is the canonical name itself
FIELD_SYNONYMS = {
    "patient_name": [
        "patient_name",
        "patient name",
        "patientname",
        "name",
        "patient",
        "full_name",
        "full name",
    ],
    "age": [
        "age",
        "patient_age",
        "patient age",
        "age_years",
        "age years",
    ],
    "gender": [
        "gender",
        "sex",
        "patient_gender",
        "patient_gender",
        "sex assigned at birth",
    ],
    "diagnosis": [
        "diagnosis",
        "primary_diagnosis",
        "primary diagnosis",
        "diagnosis_description",
        "diagnosis description",
        "clinical_diagnosis",
    ],
    "cancer_type": [
        "cancer_type",
        "cancer type",
        "cancertype",
        "tumor_type",
        "tumor type",
        "cancer_subtype",
    ],
    "cancer_stage": [
        "cancer_stage",
        "cancer stage",
        "stage",
        "tumor_stage",
        "tumor stage",
        "tnm_stage",
        "tnm stage",
        "clinical_stage",
    ],
    "biomarkers": [
        "biomarkers",
        "biomarker",
        "markers",
        "tumor_markers",
        "tumor markers",
        "immunohistochemistry",
        "ihc",
    ],
    "current_medication": [
        "current_medication",
        "current medication",
        "current_medications",
        "current medications",
        "medication",
        "medications",
        "medicine",
        "drugs",
        "drug",
        "treatment_medication",
    ],
    "previous_treatment": [
        "previous_treatment",
        "previous treatment",
        "treatment_history",
        "treatment history",
        "past_treatment",
        "prior_treatment",
        "history_of_present_illness",
    ],
    "ecog_score": [
        "ecog_score",
        "ecog score",
        "ecog",
        "performance_status",
        "performance status",
        "ps_score",
        "karnofsky",
    ],
    "follow_up_plan": [
        "follow_up_plan",
        "follow up plan",
        "follow_up",
        "followup",
        "plan",
        "follow_up_recommendations",
        "follow up recommendations",
    ],
    "next_steps": [
        "next_steps",
        "next steps",
        "recommendations",
        "plan",
        "management_plan",
        "management plan",
        "treatment_plan",
    ],
}


def find_canonical_field(raw_field_name):
    """
    Find the canonical (standard) field name for a given raw field name.
    
    How this works:
        1. We receive a field name from a document (e.g., "Tumor Stage")
        2. We check each canonical field's synonym list
        3. If a match is found, we return the canonical name (e.g., "cancer_stage")
        4. If no match is found, we return the raw name (no mapping possible)
    
    Parameters:
        raw_field_name (str): The field name as it appears in the document
    
    Returns:
        str: The canonical field name, or the original if no mapping found
    
    Python concepts used:
        - For loop with break (stop searching when found)
        - .lower() for case-insensitive matching
        - Dictionary .items() to iterate key-value pairs
        - List membership check (in)
    """
    
    # Convert to lowercase for case-insensitive matching
    raw_lower = raw_field_name.lower().strip()
    
    # Search through all canonical fields and their synonyms
    for canonical_name, synonyms in FIELD_SYNONYMS.items():
        # Check if the raw name matches any synonym for this field
        if raw_lower in synonyms:
            logger.debug(f"Mapped '{raw_field_name}' → '{canonical_name}'")
            return canonical_name
    
    # No mapping found — return the original
    logger.debug(f"No mapping found for '{raw_field_name}', keeping as-is")
    return raw_field_name


def get_all_canonical_fields():
    """
    Get a list of all canonical (standard) field names.
    
    Returns:
        list: All canonical field names
    
    This is used by the harmonizer to know which fields to expect.
    """
    return list(FIELD_SYNONYMS.keys())


def clean_extra_fields(data_dict):
    """
    Remove or rename unknown fields in a data dictionary.
    
    What this does:
        Takes a data dictionary (from LLM extraction) and:
        1. For each field, checks if it matches a canonical name
        2. If yes, keeps the field (using canonical name)
        3. If no, tries to map it using field synonyms
        4. If still no match, drops the field (with a warning)
    
    Parameters:
        data_dict (dict): The data dictionary to clean
    
    Returns:
        dict: A new dictionary with only recognized fields
    
    Python concepts used:
        - Dictionary comprehension (if we choose to use it)
        - Regular for loop with condition
        - Creating a new dictionary
    """
    
    # Get the set of known canonical field names
    canonical_fields = get_all_canonical_fields()
    
    # Create a cleaned dictionary
    cleaned = {}
    
    # Process each field in the input
    for raw_field_name, value in data_dict.items():
        # Check if this field is already a canonical name
        if raw_field_name in canonical_fields:
            cleaned[raw_field_name] = value
        else:
            # Try to find a mapping
            mapped_name = find_canonical_field(raw_field_name)
            
            if mapped_name in canonical_fields:
                # Mapping found — use the canonical name
                cleaned[mapped_name] = value
                logger.info(f"Renamed field '{raw_field_name}' to '{mapped_name}'")
            else:
                # No mapping — skip this field
                logger.warning(f"Unknown field '{raw_field_name}' skipped during harmonization")
    
    return cleaned
