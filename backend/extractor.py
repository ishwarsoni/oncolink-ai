"""
=============================================================================
extractor.py - AI-Powered Clinical Data Extraction
=============================================================================

What this module does:
    Orchestrates the complete extraction + validation + normalization pipeline:
    
    Raw text → Build prompt → Call LLM → Parse JSON → Validate → Normalize
    
    Previously (Phase 3): Raw text → Build prompt → Call LLM → Parse JSON
    Now (Phase 4):       Raw text → Build prompt → Call LLM → Parse JSON 
                                                            → Validate 
                                                            → Normalize

Why the additional steps:
    LLMs are unreliable. They might:
    - Return incomplete JSON
    - Use wrong field names
    - Return values with wrong types
    - Miss critical fields
    - Use inconsistent capitalization or abbreviations
    
    The validation and normalization steps catch ALL of these issues
    before data reaches the user interface.

=============================================================================
"""

# Import Python's JSON module for parsing
import json

# Import for parallel execution
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our custom modules
from backend.prompt_builder import build_extraction_prompt
from backend.llm_client import create_nvidia_client, call_llm

# Phase 4: Import the new validation pipeline modules
from backend.json_parser import clean_llm_response, parse_json_safely
from backend.validator import validate_clinical_data
from backend.normalizer import run_normalization_pipeline

# Import logging
import logging

# Create a logger for this module
logger = logging.getLogger(__name__)


def run_extraction(document_text):
    """
    Run the complete AI extraction pipeline on document text.
    
    Parameters:
        document_text (str): The raw text extracted from one or more
            oncology documents.
    
    How it works (NEW Phase 4 pipeline):
        1. Build prompts: Create system + user prompts for the LLM
        2. Create client: Set up the NVIDIA API connection
        3. Call LLM: Send the prompt and get a text response
        4. Parse JSON: Clean the raw response and extract JSON
        5. Validate: Check all fields against Pydantic schema
        6. Normalize: Standardize values (capitalization, abbreviations)
        7. Return result: Return the validated + normalized data
    
    Returns:
        dict: A dictionary containing:
            - "success" (bool): True if ALL steps succeeded
            - "data" (dict or None): The validated + normalized data
            - "errors" (list): All errors encountered
            - "warnings" (list): Non-critical issues
            - "raw_response" (str): The raw LLM response (for debugging)
            - "validation" (dict): Detailed validation result
            - "normalization" (dict): Detailed normalization result
    """
    
    # Initialize the result with defaults
    result = {
        "success": False,
        "data": None,
        "errors": [],
        "warnings": [],
        "raw_response": None,
        "validation": None,
        "normalization": None
    }
    
    logger.info("=" * 50)
    logger.info("Starting AI extraction pipeline")
    logger.info("=" * 50)
    
    # =========================================================================
    # Step 1: Build the prompts
    # =========================================================================
    logger.info("Step 1/6: Building prompts")
    system_prompt, user_prompt = build_extraction_prompt(document_text)
    
    # Calculate approximate token count (rough estimate: 4 chars ≈ 1 token)
    prompt_length = len(system_prompt) + len(user_prompt)
    logger.info(f"Prompt built: ~{prompt_length // 4} estimated tokens")
    
    # =========================================================================
    # Step 2: Create the NVIDIA API client
    # =========================================================================
    logger.info("Step 2/6: Creating API client")
    try:
        client = create_nvidia_client()
    except ValueError as error:
        logger.error(f"API client creation failed: {error}")
        result["errors"].append(str(error))
        return result
    
    # =========================================================================
    # Step 3: Call the LLM
    # =========================================================================
    logger.info("Step 3/6: Calling LLM")
    try:
        response_text = call_llm(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as error:
        logger.error(f"LLM API call failed: {error}")
        result["errors"].append(f"API call failed: {error}")
        return result
    
    # Check if the LLM returned an error message
    if response_text.startswith("LLM API call failed"):
        logger.error(f"LLM returned error: {response_text}")
        result["errors"].append(response_text)
        result["raw_response"] = response_text
        return result
    
    # Store the raw response for debugging
    result["raw_response"] = response_text
    logger.info(f"LLM response received: {len(response_text)} characters")
    
    # =========================================================================
    # Step 4: Parse JSON from LLM response
    # =========================================================================
    logger.info("Step 4/6: Parsing JSON")
    
    # Use the new json_parser to clean and extract JSON
    cleaned_result = clean_llm_response(response_text)
    
    if not cleaned_result["success"]:
        logger.error(f"JSON parsing failed: {cleaned_result['error']}")
        result["errors"].append(cleaned_result["error"])
        return result
    
    # Parse the cleaned JSON string into a Python dictionary
    parsed_result = parse_json_safely(cleaned_result["json_string"])
    
    if not parsed_result["success"]:
        logger.error(f"JSON parsing failed: {parsed_result['error']}")
        result["errors"].append(parsed_result["error"])
        return result
    
    # We now have a Python dictionary from the LLM's JSON
    raw_data = parsed_result["data"]
    logger.info(f"JSON parsed successfully with {len(raw_data)} top-level fields")
    
    # =========================================================================
    # Step 5: Validate against Pydantic schema
    # =========================================================================
    logger.info("Step 5/6: Validating data")
    
    validation_result = validate_clinical_data(raw_data)
    result["validation"] = validation_result
    
    if not validation_result["valid"]:
        logger.error(f"Validation failed with {len(validation_result['errors'])} error(s)")
        result["errors"].extend(validation_result["errors"])
        result["warnings"].extend(validation_result["warnings"])
        # Return validation errors but include the raw data for debugging
        result["data"] = raw_data
        return result
    
    # Validation passed!
    validated_data = validation_result["data"]
    logger.info("Validation passed successfully")
    
    # Add any warnings from validation
    result["warnings"].extend(validation_result.get("warnings", []))
    
    # =========================================================================
    # Step 6: Normalize values
    # =========================================================================
    logger.info("Step 6/6: Normalizing data")
    
    normalization_result = run_normalization_pipeline(validated_data)
    result["normalization"] = normalization_result
    
    if not normalization_result["success"]:
        logger.warning(f"Normalization had issues: {normalization_result.get('error')}")
        # Normalization is non-critical — use validated data as fallback
        result["data"] = validated_data
    else:
        # Use the normalized data
        result["data"] = normalization_result["data"]
        logger.info("Normalization complete")
    
    # =========================================================================
    # All steps completed successfully!
    # =========================================================================
    result["success"] = True
    
    logger.info("=" * 50)
    logger.info("Extraction pipeline completed successfully!")
    logger.info(f"  - Errors: {len(result['errors'])}")
    logger.info(f"  - Warnings: {len(result['warnings'])}")
    logger.info(f"  - Data fields: {len(result['data']) if result['data'] else 0}")
    logger.info("=" * 50)
    
    return result


def attempt_json_recovery(text):
    """
    Legacy function kept for backward compatibility.
    
    This function was used in Phase 3. The Phase 4 json_parser module
    now provides better cleaning. This function is kept so any code
    that still imports it won't break.
    
    Parameters:
        text (str): The raw LLM response text
    
    Returns:
        str or None: The extracted JSON string, or None if extraction fails
    """
    result = clean_llm_response(text)
    if result["success"]:
        return result["json_string"]
    return None


def extract_from_multiple_documents(processed_files):
    """
    Combine text from multiple documents and run extraction.
    
    This is a convenience function that:
    1. Takes the list of processed file results from session_state
    2. Combines them with clear document separators
    3. Runs extraction on the combined text
    
    Parameters:
        processed_files (list): List of dictionaries from document_loader.
            Each dict has: filename, text, file_type, success
    
    Returns:
        dict: The extraction result from run_extraction()
    
    Python concepts used:
        - For loop to iterate over a list
        - Building a string with f-strings
        - Only processing files that were successfully read
    """
    
    # Initialize an empty string for combined text
    combined_text = ""
    
    # Loop through each processed file
    for file_result in processed_files:
        
        # Only include files that were successfully read
        if file_result["success"]:
            # Add a clear separator with the filename
            # This helps the LLM understand which text came from which document
            combined_text += f"--- Document: {file_result['filename']} ---\n"
            combined_text += file_result["text"] + "\n\n"
    
    # Check if we have any text to process
    if len(combined_text.strip()) == 0:
        return {
            "success": False,
            "data": None,
            "errors": ["No text could be extracted from the uploaded documents."],
            "warnings": [],
            "raw_response": None,
            "validation": None,
            "normalization": None
        }
    
    # Run extraction on the combined text
    result = run_extraction(combined_text)
    
    return result


def _extract_single(file_result):
    """Helper to run extraction on a single file (used by ThreadPoolExecutor)."""
    if not file_result["success"]:
        return {
            "success": False,
            "data": None,
            "filename": file_result["filename"],
            "errors": [f"File could not be read: {file_result.get('text', 'Unknown error')}"],
            "warnings": [],
            "raw_response": None,
            "validation": None,
            "normalization": None
        }
    extraction = run_extraction(file_result["text"])
    extraction["filename"] = file_result["filename"]
    return extraction


def extract_each_document(processed_files):
    """
    Run extraction on each document in parallel, returning per-document results.
    
    This is used for harmonization and conflict detection where we need
    separate extractions from each document.
    
    Parameters:
        processed_files (list): List of dicts from document_loader.
            Each has: filename, text, file_type, success
    
    Returns:
        list: List of extraction result dicts, each with an added "filename" key
    """
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(_extract_single, fr): fr for fr in processed_files}
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    return results
