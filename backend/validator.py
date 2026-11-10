"""
validator.py - Validate LLM output against the Pydantic clinical schema.

LLMs are not databases: they return wrong field names, wrong types, extra
fields, or impossible values.  Validation happens before normalization and
before anything reaches the UI, so a malformed response cannot silently
corrupt a patient record.
"""

import logging
from typing import Any, Dict, List

from pydantic import ValidationError

from domain.clinical import BiomarkerItem, ClinicalData, create_validation_result

logger = logging.getLogger(__name__)

# Fields whose absence is worth surfacing to the user, even though the model
# accepts them as optional.
_MISSING_FIELD_WARNINGS = ("patient_name", "diagnosis", "cancer_type")


def validate_clinical_data(data_dict: Any) -> Dict[str, Any]:
    """
    Validate a parsed LLM response against ``ClinicalData``.

    Returns a dict with ``valid``, ``data``, ``errors`` and ``warnings``.
    All field errors are collected so the user can fix everything at once.
    """
    logger.info("Validation started for clinical data dictionary")

    if not isinstance(data_dict, dict):
        logger.error("Validation failed: input is not a dictionary")
        return create_validation_result(
            valid=False,
            data=None,
            errors=["Input data is not a dictionary. LLM returned unexpected format."],
        )

    if not data_dict:
        logger.warning("Validation warning: data dictionary is empty")
        return create_validation_result(
            valid=False,
            data=None,
            errors=["No data found. LLM returned an empty response."],
        )

    try:
        # Non-strict validation lets Pydantic coerce values such as "58" to 58,
        # which is required because the LLM often returns numbers as strings.
        validated_model = ClinicalData.model_validate(data_dict)
    except ValidationError as error:
        logger.error("Validation failed with %d error(s)", len(error.errors()))
        return create_validation_result(
            valid=False,
            data=None,
            errors=_format_validation_errors(error),
        )
    except Exception as error:  # defensive: never crash the extraction pipeline
        logger.error("Unexpected validation error: %s", error)
        return create_validation_result(
            valid=False,
            data=None,
            errors=[f"Unexpected validation error: {error}"],
        )

    validated_dict = validated_model.model_dump()
    logger.info("Validation passed successfully")
    return create_validation_result(
        valid=True,
        data=validated_dict,
        warnings=_collect_missing_field_warnings(validated_dict),
    )


def _format_validation_errors(error: ValidationError) -> List[str]:
    """Turn Pydantic error objects into readable, field-level messages."""
    messages = []
    for detail in error.errors():
        field_name = ".".join(str(part) for part in detail["loc"])
        message = detail["msg"]
        input_value = detail.get("input", "N/A")
        if field_name:
            messages.append(f"Field '{field_name}': {message} (got: {input_value})")
        else:
            messages.append(f"{message} (got: {input_value})")
    return messages


def _collect_missing_field_warnings(validated_data: Dict[str, Any]) -> List[str]:
    """Warn about recommended fields that the documents did not contain."""
    warnings = []
    for field_name in _MISSING_FIELD_WARNINGS:
        if validated_data.get(field_name) is None:
            warning = f"Missing recommended field: '{field_name}'"
            warnings.append(warning)
            logger.warning(warning)

    if not validated_data.get("biomarkers"):
        logger.warning("No biomarkers found in the extracted data")
    return warnings


def validate_biomarker_item(biomarker_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a single biomarker dict against ``BiomarkerItem``."""
    try:
        validated = BiomarkerItem.model_validate(biomarker_dict)
        return create_validation_result(valid=True, data=validated.model_dump())
    except ValidationError as error:
        return create_validation_result(valid=False, data=None, errors=[str(error)])
