"""
=============================================================================
harmonizer.py - Clinical Data Harmonization Engine
=============================================================================

What this module does:
    Takes multiple validated patient data objects (one per document) and
    merges them into ONE unified patient record.
    
    This is the core of Phase 5 — the "Harmonization Engine."
    
    Input: 3-5 separate patient data dicts (from different documents)
    Output: 1 unified patient data dict (with values merged from all sources)

What is Data Harmonization?
    Data harmonization is the process of combining data from different
    sources into a single, unified view. In healthcare:
    
    - Hospital A has: patient name, age, diagnosis
    - Lab B has: patient name, biomarkers, lab values
    - Clinic C has: patient name, medications, ECOG
    
    Harmonization merges all three into ONE patient record that contains
    ALL the information from ALL sources.
    
    This is different from data integration (which is about connecting
    systems) and data consolidation (which is about removing duplicates).

Why is this hard in healthcare?
    1. Different hospitals use different systems (Epic, Cerner, paper)
    2. Same patient data might be spelled differently across sources
    3. Different sources might CONTRADICT each other
    4. No single "source of truth" for all fields
    
    OncoLink solves this by:
    1. Extracting data using AI (consistent format)
    2. Validating all data against the same schema
    3. Normalizing values to standard forms
    4. Merging using field-specific strategies

Merge Strategies (per field type):
    - IDENTITY fields: patient_name, age, gender
      → Take first valid value (all should agree)
    
    - TEXT fields: diagnosis, cancer_type, cancer_stage
      → Prefer longest/most detailed value
    
    - LIST fields: biomarkers
      → Merge unique items (combine all findings)
    
    - MERGE fields: current_medication, previous_treatment
      → Combine unique strings from all sources
    
    - SINGLE fields: ecog_score
      → Take first valid number
    
    - PLAN fields: follow_up_plan, next_steps
      → Merge unique recommendations

=============================================================================
"""

# Import logging
import logging

# Import our merge utilities
from backend.merge_utils import (
    first_non_null,
    first_non_empty_string,
    prefer_longest_string,
    merge_string_lists,
    merge_unique_strings,
    merge_biomarker_lists,
    merge_numbers,
)

# Import logging
import logging

# Create a logger
logger = logging.getLogger(__name__)


def harmonize_extractions(extraction_results):
    """
    Merge multiple document extraction results into one unified record.
    
    This is the MAIN function of the harmonization engine.
    
    Parameters:
        extraction_results (list): A list of extraction result dicts.
            Each result is the output of extractor.run_extraction()
            and contains: success, data (dict), errors, warnings,
            raw_response, validation, normalization.
    
    How harmonization works:
        1. Collect all valid extraction results (skip failed ones)
        2. For each field in the schema:
           a. Collect values from ALL documents
           b. Apply the appropriate merge strategy
           c. Track which document contributed which value
        3. Return a single unified patient record
    
    Returns:
        dict: A dictionary containing:
            - "success" (bool): True if at least one document was merged
            - "data" (dict): The unified patient record
            - "documents_merged" (int): Number of documents merged
            - "field_sources" (dict): Tracks which doc each value came from
            - "warnings" (list): Any warnings during merging
    """
    
    logger.info("=" * 50)
    logger.info("Starting harmonization")
    logger.info("=" * 50)
    
    # =========================================================================
    # Step 1: Collect valid extraction data
    # =========================================================================
    # We only merge documents that were successfully extracted.
    # Failed extractions are skipped with a warning.
    valid_data_list = []
    doc_names = []
    
    for result in extraction_results:
        if result["success"] and result["data"] is not None:
            valid_data_list.append(result["data"])
            # Use the filename from raw_response or a placeholder
            doc_name = result.get("filename", "Unknown document")
            doc_names.append(doc_name)
            logger.info(f"Adding data from document: {doc_name}")
        else:
            doc_name = result.get("filename", "Unknown document")
            logger.warning(f"Skipping failed extraction for: {doc_name}")
    
    # Check if we have at least one valid extraction
    if len(valid_data_list) == 0:
        logger.error("No valid extraction data to harmonize")
        return {
            "success": False,
            "data": None,
            "documents_merged": 0,
            "field_sources": {},
            "warnings": ["No valid extraction data available to harmonize."]
        }
    
    logger.info(f"Harmonizing {len(valid_data_list)} document(s)")
    
    # =========================================================================
    # Step 2: Define merge strategies for each field
    # =========================================================================
    # Each field type needs a different merge approach.
    # This dictionary maps field names to their merge strategy.
    # The strategy is a function that takes a list of values and returns
    # the merged result.
    # =========================================================================
    
    # Build the unified record and track field sources
    unified = {}
    field_sources = {}
    warnings = []
    
    # =========================================================================
    # IDENTITY FIELDS — Take first valid value
    # =========================================================================
    # These fields should have the same value across all documents.
    # We take the first non-None value we find.
    identity_fields = ["patient_name", "age", "gender"]
    merge_identity(unified, field_sources, valid_data_list, doc_names, identity_fields)
    
    # =========================================================================
    # TEXT FIELDS — Prefer longest/most detailed
    # =========================================================================
    # For diagnosis and cancer type, a longer value is usually more detailed
    # and therefore better (contains more clinical information).
    text_fields = ["diagnosis", "cancer_type", "cancer_stage"]
    merge_text(unified, field_sources, valid_data_list, doc_names, text_fields)
    
    # =========================================================================
    # BIOMARKERS — Merge unique items
    # =========================================================================
    # Each document might contribute different biomarkers.
    # We merge them all, deduplicating by name.
    merge_biomarkers(unified, field_sources, valid_data_list, doc_names)
    
    # =========================================================================
    # MERGE STRING FIELDS — Combine unique strings
    # =========================================================================
    # Medications and treatments might differ across documents.
    # We combine all unique items into one string.
    merge_string_fields(unified, field_sources, valid_data_list, doc_names)
    
    # =========================================================================
    # SINGLE VALUE FIELDS — Take first valid
    # =========================================================================
    single_fields = ["ecog_score"]
    merge_single_values(unified, field_sources, valid_data_list, doc_names, single_fields)
    
    # =========================================================================
    # PLAN FIELDS — Merge unique recommendations
    # =========================================================================
    plan_fields = ["follow_up_plan", "next_steps"]
    merge_plan_fields(unified, field_sources, valid_data_list, doc_names, plan_fields)
    
    # =========================================================================
    # Step 3: Return the unified record
    # =========================================================================
    
    logger.info(f"Harmonization complete: {len(valid_data_list)} documents merged")
    logger.info(f"Unified record has {len(unified)} fields")
    
    return {
        "success": True,
        "data": unified,
        "documents_merged": len(valid_data_list),
        "field_sources": field_sources,
        "warnings": warnings
    }


# =============================================================================
# Merge Strategy Functions
# =============================================================================
# Each function implements ONE merge strategy for a group of fields.
# Keeping them separate makes the code easy to test and modify.
# =============================================================================


def merge_identity(unified, field_sources, valid_data_list, doc_names, fields):
    """
    Merge identity fields by taking the first non-null value.
    
    For fields like patient_name, age, and gender, all documents should
    have the same value. We take the first valid one we find.
    
    Parameters:
        unified (dict): The unified record being built (modified in place)
        field_sources (dict): Source tracking (modified in place)
        valid_data_list (list): List of validated data dicts
        doc_names (list): Corresponding document names
        fields (list): Field names to merge with this strategy
    """
    for field_name in fields:
        # Collect values for this field from all documents
        values = []
        for data in valid_data_list:
            values.append(data.get(field_name))
        
        # Apply the merge strategy
        merged = first_non_null(values)
        
        if merged is not None:
            unified[field_name] = merged
            # Track which document contributed this value
            for idx, val in enumerate(values):
                if val is not None and idx < len(doc_names):
                    field_sources[field_name] = doc_names[idx]
                    break
            logger.debug(f"Identity merge '{field_name}': '{merged}' from '{field_sources.get(field_name)}'")


def merge_text(unified, field_sources, valid_data_list, doc_names, fields):
    """
    Merge text fields by preferring the longest/most detailed value.
    
    For diagnosis and cancer_type, a more detailed value is better.
    "Invasive ductal carcinoma of the left breast" has more information
    than just "Invasive ductal carcinoma."
    
    Parameters:
        unified (dict): The unified record being built
        field_sources (dict): Source tracking
        valid_data_list (list): List of validated data dicts
        doc_names (list): Corresponding document names
        fields (list): Field names to merge
    """
    for field_name in fields:
        values = []
        for data in valid_data_list:
            values.append(data.get(field_name))
        
        # Prefer the longest string (usually has more detail)
        merged = prefer_longest_string(values)
        
        # Fallback to first non-empty if longest fails
        if merged is None:
            merged = first_non_empty_string(values)
        
        if merged is not None:
            unified[field_name] = merged
            # Track source
            for idx, val in enumerate(values):
                if (val and isinstance(val, str) and val.strip() == merged and idx < len(doc_names)):
                    field_sources[field_name] = doc_names[idx]
                    break
            logger.debug(f"Text merge '{field_name}': '{merged[:50]}...' from '{field_sources.get(field_name)}'")


def merge_biomarkers(unified, field_sources, valid_data_list, doc_names):
    """
    Merge biomarker lists from all documents.
    
    Each document might report different biomarkers:
    - Pathology: ER+, PR+, HER2-
    - Lab: Ki-67 25%
    
    The merged list should include ALL unique biomarkers.
    
    Parameters:
        unified (dict): The unified record being built
        field_sources (dict): Source tracking
        valid_data_list (list): List of validated data dicts
        doc_names (list): Corresponding document names
    """
    # Collect biomarker lists from all documents
    biomarker_lists = []
    for data in valid_data_list:
        bm_list = data.get("biomarkers", [])
        if isinstance(bm_list, list) and len(bm_list) > 0:
            biomarker_lists.append(bm_list)
    
    if len(biomarker_lists) > 0:
        merged = merge_biomarker_lists(biomarker_lists)
        unified["biomarkers"] = merged
        
        # Track sources
        source_docs = []
        for idx, data in enumerate(valid_data_list):
            bm_list = data.get("biomarkers", [])
            if isinstance(bm_list, list) and len(bm_list) > 0 and idx < len(doc_names):
                source_docs.append(doc_names[idx])
        
        if len(source_docs) > 0:
            field_sources["biomarkers"] = "; ".join(source_docs)
        
        logger.debug(f"Biomarker merge: {len(merged)} unique biomarkers from {len(source_docs)} document(s)")
    else:
        unified["biomarkers"] = []


def merge_string_fields(unified, field_sources, valid_data_list, doc_names):
    """
    Merge text fields by combining unique strings from all documents.
    
    For medications and treatments, each document might list different items.
    We combine all unique items into a single string separated by semicolons.
    
    Parameters:
        unified (dict): The unified record being built
        field_sources (dict): Source tracking
        valid_data_list (list): List of validated data dicts
        doc_names (list): Corresponding document names
    """
    fields = ["current_medication", "previous_treatment", "adverse_events"]
    
    for field_name in fields:
        values = []
        for data in valid_data_list:
            values.append(data.get(field_name))
        
        merged = merge_unique_strings(values)
        
        if merged is not None:
            unified[field_name] = merged
            # Track sources
            sources = []
            for idx, val in enumerate(values):
                if val and isinstance(val, str) and val.strip() and idx < len(doc_names):
                    sources.append(doc_names[idx])
            if len(sources) > 0:
                field_sources[field_name] = "; ".join(sources)
            logger.debug(f"String merge '{field_name}': {len(merged)} characters from {len(sources)} source(s)")


def merge_single_values(unified, field_sources, valid_data_list, doc_names, fields):
    """
    Merge single numeric values by taking the first valid one.
    
    For fields like ecog_score, all documents should agree.
    We take the first valid non-None value.
    
    Parameters:
        unified (dict): The unified record being built
        field_sources (dict): Source tracking
        valid_data_list (list): List of validated data dicts
        doc_names (list): Corresponding document names
        fields (list): Field names to merge
    """
    for field_name in fields:
        values = []
        for data in valid_data_list:
            values.append(data.get(field_name))
        
        merged = merge_numbers(values)
        
        if merged is not None:
            unified[field_name] = merged
            for idx, val in enumerate(values):
                if val is not None and idx < len(doc_names):
                    field_sources[field_name] = doc_names[idx]
                    break
            logger.debug(f"Single value merge '{field_name}': '{merged}'")


def merge_plan_fields(unified, field_sources, valid_data_list, doc_names, fields):
    for field_name in fields:
        values = []
        for data in valid_data_list:
            values.append(data.get(field_name))

        merged = merge_unique_strings(values)

        if merged is not None:
            if field_name == "next_steps":
                items = [i.strip() for i in merged.split(";")]
                has_continue = any(i.lower().startswith("continue") for i in items if i)
                if has_continue:
                    items = [i for i in items if not (i and i.lower().startswith("start"))]
                merged = "; ".join(items) if items else merged

            unified[field_name] = merged
            sources = []
            for idx, val in enumerate(values):
                if val and isinstance(val, str) and val.strip() and idx < len(doc_names):
                    sources.append(doc_names[idx])
            if len(sources) > 0:
                field_sources[field_name] = "; ".join(sources)
            logger.debug(f"Plan merge '{field_name}': {len(merged)} characters")
