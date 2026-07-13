"""
=============================================================================
validator.py - Validate Clinical Data Against Pydantic Schema
=============================================================================

What this module does:
    Takes a dictionary of clinical data (parsed from LLM JSON response)
    and validates it against our Pydantic ClinicalData model.
    
    Validation checks:
    1. Field names match the Pydantic model (no unknown fields)
    2. Field types match (strings are strings, ints are ints, etc.)
    3. Values satisfy constraints (age >= 0, ECOG 0-5, etc.)
    4. Biomarker list items are valid BiomarkerItem objects

Why validate LLM output:
    LLMs are NOT databases. They don't guarantee consistent output.
    Common problems:
    - Wrong field names: "Patient_Name" instead of "patient_name"
    - Wrong types: age as "fifty eight" instead of 58
    - Missing fields: No diagnosis field at all
    - Garbage values: ecog_score = 99 (invalid, max is 5)
    - Extra fields: JSON with fields we never asked for
    
    Validation catches ALL of these before the data reaches the UI.

What is a ValidationError:
    When Pydantic receives data that doesn't match the model, it raises
    a ValidationError. This error contains DETAILED information about
    exactly which fields failed and why.
    
    Example:
        Field: age
        Error: Input should be a valid integer
        Value received: "fifty eight"

=============================================================================
"""

# Import Pydantic's ValidationError
# This is the exception raised when data doesn't match the model
from pydantic import ValidationError

# Import our Pydantic models
from backend.schema import ClinicalData, BiomarkerItem, create_validation_result

# Import our logging module
import logging

# Create a logger for this module
# The logger records validation events so we can debug issues later
logger = logging.getLogger(__name__)


def validate_clinical_data(data_dict):
    """
    Validate a dictionary of clinical data against the Pydantic model.
    
    Parameters:
        data_dict (dict): The clinical data dictionary to validate.
            This comes from parsing the LLM's JSON response.
    
    How validation works:
        1. We try to create a ClinicalData object from the dictionary.
        2. Pydantic automatically checks all fields, types, and constraints.
        3. If everything is valid, we return the validated data.
        4. If validation fails, we collect ALL errors and return them.
           (Not just the first error — we want to see all issues at once.)
    
    Returns:
        dict: A validation result with:
            - "valid" (bool): True if all checks passed
            - "data" (dict or None): Validated data as a dict (if valid)
            - "errors" (list): Critical error messages (field-level)
            - "warnings" (list): Non-critical issues
    
    Python concepts used:
        - try/except with specific exception type (ValidationError)
        - Pydantic's model_validate() method
        - Pydantic's model_dump() method (convert back to dict)
        - Iterating over error details
        - Logging with different severity levels
    """
    
    # Log that validation has started
    logger.info(f"Validation started for clinical data dictionary")
    
    # Step 1: Check if the input is a dictionary
    if not isinstance(data_dict, dict):
        logger.error("Validation failed: input is not a dictionary")
        return create_validation_result(
            valid=False,
            data=None,
            errors=["Input data is not a dictionary. LLM returned unexpected format."]
        )
    
    # Step 2: Check if the dictionary is empty
    if len(data_dict) == 0:
        logger.warning("Validation warning: data dictionary is empty")
        return create_validation_result(
            valid=False,
            data=None,
            errors=["No data found. LLM returned an empty response."]
        )
    
    # =========================================================================
    # Step 3: Validate using Pydantic
    # =========================================================================
    # We use ClinicalData.model_validate() to check all fields at once.
    # If validation fails, Pydantic raises ValidationError with details.
    #
    # Why model_validate() instead of ClinicalData(**data_dict)?
    #   model_validate() is the Pydantic v2 way. It provides better
    #   error messages and supports strict mode.
    #
    # Why NOT strict mode:
    #   Strict mode means "age must be int, not string '58'". 
    #   But LLMs often return "58" as a string. We want to be lenient
    #   and let Pydantic try to coerce types automatically.
    # =========================================================================
    
    try:
        # Try to create a validated ClinicalData object
        # Pydantic automatically:
        # - Checks field names match the model
        # - Converts types (e.g., "58" string to 58 int)
        # - Checks constraints (e.g., ECOG 0-5)
        # - Rejects unknown fields (e.g., "favorite_color")
        validated_model = ClinicalData.model_validate(data_dict)
        
        # Validation succeeded!
        logger.info("Validation passed successfully")
        
        # Convert the validated model back to a dictionary
        # model_dump() converts Pydantic objects to plain Python dicts
        # We use it so the rest of the app works with regular dicts
        validated_dict = validated_model.model_dump()
        
        # Check for non-critical warnings
        warnings = []
        
        # Warning: Required fields that are missing
        required_fields = ["patient_name", "diagnosis", "cancer_type"]
        for field_name in required_fields:
            if validated_dict.get(field_name) is None:
                warning_msg = f"Missing recommended field: '{field_name}'"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        # Warning: Empty biomarkers list
        if len(validated_dict.get("biomarkers", [])) == 0:
            logger.warning("No biomarkers found in the extracted data")
        
        return create_validation_result(
            valid=True,
            data=validated_dict,
            errors=[],
            warnings=warnings
        )
    
    except ValidationError as error:
        # =====================================================================
        # Validation Failed — Collect All Errors
        # =====================================================================
        # Pydantic's ValidationError contains a list of errors.
        # Each error has:
        #   - loc: The field name (e.g., ("ecog_score",))
        #   - msg: Human-readable error message
        #   - type: Error type (e.g., "int_parsing", "greater_than_equal")
        # 
        # We collect ALL errors (not just the first one) so the user
        # can fix everything at once.
        # =====================================================================
        
        logger.error(f"Validation failed with {len(error.errors())} error(s)")
        
        collected_errors = []
        field_warnings = []
        
        # Loop through each individual error from Pydantic
        for err in error.errors():
            # Get the field name from the error location
            # err["loc"] is a tuple like ("ecog_score",)
            # We join with dots in case of nested fields
            field_name = ".".join(str(loc) for loc in err["loc"])
            
            # Get the error message
            message = err["msg"]
            
            # Get the input value that caused the error (if available)
            input_value = err.get("input", "N/A")
            
            # Create a user-friendly error message
            if field_name:
                full_message = f"Field '{field_name}': {message} (got: {input_value})"
            else:
                full_message = f"{message} (got: {input_value})"
            
            # Decide if this is a critical error or a warning
            # Type errors (wrong type) and constraint errors are critical
            # Missing values for optional fields are warnings
            collected_errors.append(full_message)
        
        return create_validation_result(
            valid=False,
            data=None,
            errors=collected_errors,
            warnings=field_warnings
        )
    
    except Exception as error:
        # Catch any unexpected errors (shouldn't happen, but defensive)
        logger.error(f"Unexpected validation error: {error}")
        return create_validation_result(
            valid=False,
            data=None,
            errors=[f"Unexpected validation error: {error}"]
        )


def validate_biomarker_item(biomarker_dict):
    """
    Validate a single biomarker dictionary against BiomarkerItem model.
    
    Parameters:
        biomarker_dict (dict): A dictionary with name, value, status
    
    Returns:
        dict: Validation result with success status
    
    This is a helper function used when we want to validate individual
    biomarkers rather than the whole ClinicalData model at once.
    """
    try:
        validated = BiomarkerItem.model_validate(biomarker_dict)
        return create_validation_result(
            valid=True,
            data=validated.model_dump()
        )
    except ValidationError as error:
        return create_validation_result(
            valid=False,
            data=None,
            errors=[str(error)]
        )
