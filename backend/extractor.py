"""
extractor.py - Orchestrate extraction, validation, and normalization.

Pipeline for one body of text:

    build prompt -> call LLM -> parse JSON -> validate -> normalize

The LLM is treated as an unreliable source: a malformed response is reported
as an error instead of reaching the UI as clinical data.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from backend.json_parser import clean_llm_response, parse_json_safely
from backend.llm_client import call_llm, create_nvidia_client
from backend.normalizer import run_normalization_pipeline
from backend.prompt_builder import build_extraction_prompt
from backend.validator import validate_clinical_data

logger = logging.getLogger(__name__)

_MAX_EXTRACTION_WORKERS = 5


def _new_extraction_result() -> Dict[str, Any]:
    """Return the default result shape for the extraction pipeline."""
    return {
        "success": False,
        "data": None,
        "errors": [],
        "warnings": [],
        "raw_response": None,
        "validation": None,
        "normalization": None,
    }


def run_extraction(document_text: str) -> Dict[str, Any]:
    """
    Run the complete extraction pipeline on document text.

    Returns ``success``, ``data``, ``errors``, ``warnings``, ``raw_response``,
    and the detailed ``validation`` / ``normalization`` results.
    """
    result = _new_extraction_result()
    logger.info("Starting AI extraction pipeline")

    logger.info("Step 1/6: Building prompts")
    system_prompt, user_prompt = build_extraction_prompt(document_text)
    logger.info(
        "Prompt built: ~%d estimated tokens",
        (len(system_prompt) + len(user_prompt)) // 4,
    )

    client = _create_client(result)
    if client is None:
        return result

    response_text = _call_extraction_llm(client, system_prompt, user_prompt, result)
    if response_text is None:
        return result

    raw_data = _parse_extraction_response(response_text, result)
    if raw_data is None:
        return result

    validated_data = _validate_extraction(raw_data, result)
    if validated_data is None:
        return result

    _normalize_extraction(validated_data, result)
    result["success"] = True

    logger.info(
        "Extraction complete: %d error(s), %d warning(s), %d field(s)",
        len(result["errors"]),
        len(result["warnings"]),
        len(result["data"] or {}),
    )
    return result


def _create_client(result: Dict[str, Any]):
    """Create the NVIDIA client, recording a configuration error on failure."""
    logger.info("Step 2/6: Creating API client")
    try:
        return create_nvidia_client()
    except ValueError as error:
        logger.error("API client creation failed: %s", error)
        result["errors"].append(str(error))
        return None


def _call_extraction_llm(
    client, system_prompt: str, user_prompt: str, result: Dict[str, Any]
) -> Optional[str]:
    """Call the LLM and return its text, or ``None`` when the call failed."""
    logger.info("Step 3/6: Calling LLM")
    try:
        response_text = call_llm(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
    except Exception as error:  # defensive: the UI must show a readable error
        logger.error("LLM API call failed: %s", error)
        result["errors"].append(f"API call failed: {error}")
        return None

    # The client reports transport failures as a text prefix rather than
    # raising, so check it explicitly.
    if response_text.startswith("LLM API call failed"):
        logger.error("LLM returned error: %s", response_text)
        result["errors"].append(response_text)
        result["raw_response"] = response_text
        return None

    result["raw_response"] = response_text
    logger.info("LLM response received: %d characters", len(response_text))
    return response_text


def _parse_extraction_response(
    response_text: str, result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Clean and parse the JSON object returned by the LLM."""
    logger.info("Step 4/6: Parsing JSON")

    cleaned = clean_llm_response(response_text)
    if not cleaned["success"]:
        logger.error("JSON parsing failed: %s", cleaned["error"])
        result["errors"].append(cleaned["error"])
        return None

    parsed = parse_json_safely(cleaned["json_string"])
    if not parsed["success"]:
        logger.error("JSON parsing failed: %s", parsed["error"])
        result["errors"].append(parsed["error"])
        return None

    raw_data = parsed["data"]
    logger.info("JSON parsed successfully with %d top-level fields", len(raw_data))
    return raw_data


def _validate_extraction(
    raw_data: Dict[str, Any], result: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Validate against the Pydantic schema and return the validated dict."""
    logger.info("Step 5/6: Validating data")

    validation_result = validate_clinical_data(raw_data)
    result["validation"] = validation_result

    if not validation_result["valid"]:
        logger.error(
            "Validation failed with %d error(s)", len(validation_result["errors"])
        )
        result["errors"].extend(validation_result["errors"])
        result["warnings"].extend(validation_result["warnings"])
        # Keep the raw payload so the debug view can show what the LLM sent.
        result["data"] = raw_data
        return None

    result["warnings"].extend(validation_result.get("warnings", []))
    return validation_result["data"]


def _normalize_extraction(validated_data: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Normalize validated values, falling back to validated data on failure."""
    logger.info("Step 6/6: Normalizing data")

    normalization_result = run_normalization_pipeline(validated_data)
    result["normalization"] = normalization_result

    if normalization_result["success"]:
        result["data"] = normalization_result["data"]
        logger.info("Normalization complete")
    else:
        # Normalization is cosmetic; validated data is still usable.
        logger.warning("Normalization had issues: %s", normalization_result.get("error"))
        result["data"] = validated_data


def attempt_json_recovery(text: str) -> Optional[str]:
    """
    Return the JSON substring of an LLM response, or ``None``.

    Kept as a public helper for callers and tests that only need the cleaned
    JSON string rather than the full extraction pipeline.
    """
    result = clean_llm_response(text)
    return result["json_string"] if result["success"] else None


def extract_from_multiple_documents(processed_files: List[Dict]) -> Dict[str, Any]:
    """Combine all successfully read documents and extract once."""
    combined_text = "".join(
        f"--- Document: {item['filename']} ---\n{item['text']}\n\n"
        for item in processed_files
        if item["success"]
    )

    if not combined_text.strip():
        result = _new_extraction_result()
        result["errors"].append("No text could be extracted from the uploaded documents.")
        return result

    return run_extraction(combined_text)


def extract_each_document(processed_files: List[Dict]) -> List[Dict[str, Any]]:
    """
    Extract every document independently, in parallel.

    Per-document results feed harmonization and conflict detection, which need
    to know which document supplied each value.  Results are returned in the
    same order as ``processed_files`` so "first value wins" stays predictable.
    """
    results: List[Optional[Dict[str, Any]]] = [None] * len(processed_files)

    with ThreadPoolExecutor(max_workers=_MAX_EXTRACTION_WORKERS) as executor:
        futures = {
            executor.submit(_extract_single, item): index
            for index, item in enumerate(processed_files)
        }
        for future in as_completed(futures):
            results[futures[future]] = future.result()

    return [result for result in results if result is not None]


def _extract_single(file_result: Dict) -> Dict[str, Any]:
    """Extract one processed file and tag the result with its filename."""
    if not file_result["success"]:
        result = _new_extraction_result()
        result["filename"] = file_result["filename"]
        result["errors"].append(
            f"File could not be read: {file_result.get('text', 'Unknown error')}"
        )
        return result

    extraction = run_extraction(file_result["text"])
    extraction["filename"] = file_result["filename"]
    return extraction
