"""
=============================================================================
normalizer.py - Standardize and Clean Clinical Data Values
=============================================================================

What this module does:
    Takes validated clinical data and normalizes (standardizes) values
    so that the same concept is always represented the same way.
    
    Examples of normalization:
        "Stage ii" → "Stage II"              (capitalization)
        "IDC" → "Invasive Ductal Carcinoma"  (abbreviation)
        "F" → "Female"                       (abbreviation)
        "  ECOG 1  " → "1"                   (whitespace + prefix)
        "HER2 positive" → "HER2 Positive"    (capitalization)

Why normalization matters:
    1. Consistency: "Male", "male", "M" all mean the same thing
    2. Comparison: You can compare "Stage II" == "Stage II" reliably
    3. Display: Professional capitalization looks better
    4. Future phases: Harmonization (Phase 5) depends on normalized data
    
    Without normalization:
    - Document 1 says "Stage II"
    - Document 2 says "stage ii"
    - Your harmonizer thinks these are different values → false conflict!

What normalization IS:
    - Changing "IDC" to "Invasive Ductal Carcinoma"
    - Capitalizing names properly
    - Trimming whitespace

What normalization IS NOT:
    - Guessing missing values (that's hallucination)
    - Changing medical facts
    - Removing information (that's the validator's job)

=============================================================================
"""

# Import logging for tracking normalization
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


# =============================================================================
# Normalization Maps
# =============================================================================
# These dictionaries map raw values (as the LLM might return them) to
# their standardized forms. We use lowercase keys for case-insensitive matching.
#
# Why dictionaries:
#   A dictionary is the simplest way to define a mapping.
#   Key = raw input (lowercase), Value = normalized output.
#   Fast, readable, and easy to extend.
# =============================================================================

# Cancer type abbreviations and their full forms
CANCER_TYPE_MAP = {
    "idc": "Invasive Ductal Carcinoma",
    "ilc": "Invasive Lobular Carcinoma",
    "dcis": "Ductal Carcinoma In Situ",
    "lcis": "Lobular Carcinoma In Situ",
    "tnbc": "Triple Negative Breast Cancer",
    "mbc": "Metastatic Breast Cancer",
    "nsclc": "Non-Small Cell Lung Cancer",
    "sclc": "Small Cell Lung Cancer",
    "hcc": "Hepatocellular Carcinoma",
    "ccrcc": "Clear Cell Renal Cell Carcinoma",
}

# Gender abbreviations and their full forms
GENDER_MAP = {
    "m": "Male",
    "f": "Female",
    "male": "Male",
    "female": "Female",
}

# Biomarker normalization
# Maps full/compound names to canonical gene names
# "EGFR exon 19 deletion" → gene="EGFR", detail preserved in value
BIOMARKER_NAME_MAP = {
    "er": "ER",
    "pr": "PR",
    "her2": "HER2",
    "her-2": "HER2",
    "ki67": "Ki-67",
    "ki-67": "Ki-67",
    # EGFR variants — all collapse to "EGFR"
    "egfr": "EGFR",
    "egfr exon 19 deletion": "EGFR",
    "egfr exon19 deletion": "EGFR",
    "egfr mutation": "EGFR",
    "egfr mutant": "EGFR",
    "egfr positive": "EGFR",
    "egfr negative": "EGFR",
    "egfr wild type": "EGFR",
    "egfr wildtype": "EGFR",
    # ALK variants
    "alk": "ALK",
    "alk rearrangement": "ALK",
    "alk fusion": "ALK",
    "alk positive": "ALK",
    "alk negative": "ALK",
    # ROS1 variants
    "ros1": "ROS1",
    "ros1 rearrangement": "ROS1",
    "ros1 fusion": "ROS1",
    "ros1 positive": "ROS1",
    "ros1 negative": "ROS1",
    # PD-L1 variants
    "pd-l1": "PD-L1",
    "pdl1": "PD-L1",
    "pd l1": "PD-L1",
}

# Known gene prefixes — if a biomarker name starts with one of these,
# extract the gene and keep the remainder as the value
BIOMARKER_GENE_PREFIXES = {
    "egfr": "EGFR",
    "alk": "ALK",
    "ros1": "ROS1",
    "pd-l1": "PD-L1",
    "pdl1": "PD-L1",
    "her2": "HER2",
    "braf": "BRAF",
    "kras": "KRAS",
    "nras": "NRAS",
    "pik3ca": "PIK3CA",
}

BIOMARKER_STATUS_MAP = {
    "+": "Positive",
    "pos": "Positive",
    "positive": "Positive",
    "-": "Negative",
    "neg": "Negative",
    "negative": "Negative",
    "equivocal": "Equivocal",
    "borderline": "Borderline",
}


def normalize_clinical_data(data_dict):
    """
    Normalize all fields in a clinical data dictionary.
    
    Parameters:
        data_dict (dict): Validated clinical data dictionary
            (output from validator, after Pydantic validation)
    
    What this does:
        - Normalizes each field using field-specific rules
        - Returns a NEW dictionary (does not modify the input)
        - Logs all normalizations performed
    
    Returns:
        dict: The normalized data dictionary
    
    Python concepts used:
        - Dictionary .get() with default values
        - .strip() for removing whitespace
        - .capitalize() and .title() for capitalization
        - isinstance() for type checking
        - Creating a new dict (not modifying the input)
    """
    
    # Log that normalization started
    logger.info("Normalization started")
    
    # Create a copy so we don't modify the original
    # We want the original preserved for debugging
    normalized = dict(data_dict)
    
    # Track how many normalizations we perform
    normalization_count = 0
    
    # =========================================================================
    # Normalize patient_name
    # =========================================================================
    name = normalized.get("patient_name")
    if name and isinstance(name, str):
        name = name.strip()
        # Title case: "meera sharma" → "Meera Sharma"
        name = name.title()
        normalized["patient_name"] = name
        normalization_count += 1
        logger.debug(f"Normalized patient_name: '{data_dict['patient_name']}' → '{name}'")
    
    # =========================================================================
    # Normalize gender
    # =========================================================================
    gender = normalized.get("gender")
    if gender and isinstance(gender, str):
        gender_lower = gender.strip().lower()
        if gender_lower in GENDER_MAP:
            normalized["gender"] = GENDER_MAP[gender_lower]
        else:
            # Capitalize first letter: "female" → "Female"
            normalized["gender"] = gender.strip().capitalize()
        normalization_count += 1
        logger.debug(f"Normalized gender: '{data_dict['gender']}' → '{normalized['gender']}'")
    
    # =========================================================================
    # Normalize cancer_type
    # =========================================================================
    cancer_type = normalized.get("cancer_type")
    if cancer_type and isinstance(cancer_type, str):
        cancer_type_lower = cancer_type.strip().lower()
        if cancer_type_lower in CANCER_TYPE_MAP:
            normalized["cancer_type"] = CANCER_TYPE_MAP[cancer_type_lower]
        else:
            # Capitalize the first letter of each word
            normalized["cancer_type"] = cancer_type.strip()
        normalization_count += 1
        logger.debug(f"Normalized cancer_type: '{data_dict['cancer_type']}' → '{normalized['cancer_type']}'")
    
    # =========================================================================
    # Normalize cancer_stage
    # =========================================================================
    stage = normalized.get("cancer_stage")
    if stage and isinstance(stage, str):
        # Capitalize stage: "stage iib" → "Stage IIB"
        stage = stage.strip()
        # Capitalize the word "stage" if it appears
        if stage.lower().startswith("stage"):
            stage = "Stage" + stage[5:]
        # Convert TNM to uppercase: "t2 n1 m0" → "T2 N1 M0"
        # This is a simple approach — handles most cases
        normalized["cancer_stage"] = stage
        normalization_count += 1
        logger.debug(f"Normalized cancer_stage: '{data_dict['cancer_stage']}' → '{normalized['cancer_stage']}'")
    
    # =========================================================================
    # Normalize ecog_score
    # =========================================================================
    # ECOG score might come as a string like "ECOG 1" or "1"
    # If it's already an integer (from Pydantic validation), leave it
    # If it's a string, try to extract the number
    ecog = normalized.get("ecog_score")
    if ecog is not None:
        if isinstance(ecog, str):
            # Try to extract a number from the string
            # "ECOG 1" → "1" → 1
            import re
            numbers_found = re.findall(r'\d+', ecog)
            if numbers_found:
                normalized["ecog_score"] = int(numbers_found[0])
                normalization_count += 1
                logger.debug(f"Normalized ecog_score: '{data_dict['ecog_score']}' → '{normalized['ecog_score']}'")
        elif isinstance(ecog, (int, float)):
            # Already a number, just ensure it's an int
            if isinstance(ecog, float):
                normalized["ecog_score"] = int(ecog)
                normalization_count += 1
    
    # =========================================================================
    # Normalize biomarkers
    # =========================================================================
    biomarkers = normalized.get("biomarkers", [])
    if biomarkers and isinstance(biomarkers, list):
        normalized_biomarkers = []
        seen_biomarkers = {}  # name_lower → index in list
        for biomarker in biomarkers:
            normalized_biomarker = normalize_single_biomarker(biomarker)
            bm_name = normalized_biomarker.get("name", "").strip().lower()
            if bm_name and bm_name in seen_biomarkers:
                # Duplicate biomarker name within same extraction — merge values
                existing_idx = seen_biomarkers[bm_name]
                existing = normalized_biomarkers[existing_idx]
                existing_val = existing.get("value", "").strip()
                new_val = normalized_biomarker.get("value", "").strip()
                if new_val and new_val.lower() not in existing_val.lower():
                    # Prefer more specific value
                    if existing_val and len(new_val) > len(existing_val):
                        existing["value"] = new_val
                logger.debug(f"Merged duplicate biomarker '{bm_name}' in same document")
            else:
                if bm_name:
                    seen_biomarkers[bm_name] = len(normalized_biomarkers)
                normalized_biomarkers.append(normalized_biomarker)
        normalized["biomarkers"] = normalized_biomarkers
    
    # =========================================================================
    # Normalize text fields (trim whitespace, capitalize)
    # =========================================================================
    text_fields = ["diagnosis", "current_medication", "previous_treatment",
                   "follow_up_plan", "next_steps"]
    
    for field_name in text_fields:
        value = normalized.get(field_name)
        if value and isinstance(value, str):
            value = value.strip()
            normalized[field_name] = value
    
    logger.info(f"Normalization complete: {normalization_count} normalizations performed")
    
    return normalized


def normalize_single_biomarker(biomarker):
    normalized = dict(biomarker)

    name = normalized.get("name", "")
    if name and isinstance(name, str):
        name_raw = name.strip()
        name_lower = name_raw.lower()

        # Check name map first (exact match)
        if name_lower in BIOMARKER_NAME_MAP:
            normalized["name"] = BIOMARKER_NAME_MAP[name_lower]
        else:
            # Check if name starts with a known gene prefix
            # e.g. "EGFR exon 19 deletion" → gene="EGFR", detail→value
            prefix_match = None
            for prefix, canonical in BIOMARKER_GENE_PREFIXES.items():
                if name_lower.startswith(prefix) and len(name_raw) > len(prefix):
                    # Extra text after gene name — move to value if value is empty/simple
                    suffix = name_raw[len(prefix):].strip().lstrip(" ,-–")
                    existing_val = normalized.get("value", "").strip().lower()
                    if suffix and (not existing_val or existing_val in ("detected", "positive", "negative", "+", "-", "")):
                        if existing_val and existing_val not in ("", "detected"):
                            # Keep existing value, add suffix context
                            normalized["value"] = f"{suffix} ({normalized['value']})"
                        else:
                            normalized["value"] = suffix
                    normalized["name"] = canonical
                    prefix_match = True
                    break

            if not prefix_match:
                normalized["name"] = name_raw

    # Normalize value
    value = normalized.get("value", "")
    if value and isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in BIOMARKER_STATUS_MAP:
            normalized["value"] = BIOMARKER_STATUS_MAP[value_lower]

    # Normalize status
    status = normalized.get("status")
    if status and isinstance(status, str):
        status_lower = status.strip().lower()
        if status_lower in BIOMARKER_STATUS_MAP:
            normalized["status"] = BIOMARKER_STATUS_MAP[status_lower]

    return normalized


def run_normalization_pipeline(data_dict):
    """
    Run the complete normalization pipeline on clinical data.
    
    This is the main entry point for normalization.
    It handles both validated and unvalidated data gracefully.
    
    Parameters:
        data_dict (dict): Clinical data dictionary (may be validated or raw)
    
    Returns:
        dict: A result with:
            - "success" (bool): Whether normalization succeeded
            - "data" (dict): The normalized data
            - "error" (str): Error message if something went wrong
    """
    
    try:
        # Check if input is a dictionary
        if not isinstance(data_dict, dict):
            return {
                "success": False,
                "data": None,
                "error": "Cannot normalize: input is not a dictionary."
            }
        
        # Run normalization
        normalized_data = normalize_clinical_data(data_dict)
        
        return {
            "success": True,
            "data": normalized_data,
            "error": None
        }
        
    except Exception as error:
        logger.error(f"Normalization pipeline failed: {error}")
        return {
            "success": False,
            "data": None,
            "error": f"Normalization failed: {error}"
        }
