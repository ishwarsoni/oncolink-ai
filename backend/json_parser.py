"""
json_parser.py - Recover a JSON object from raw LLM output.

LLMs wrap JSON in markdown fences, add explanations, or leave trailing
commas even when asked not to.  These helpers isolate the JSON object so the
rest of the pipeline can assume parseable input.
"""

import json
from typing import Any, Dict, Optional


def clean_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Extract the first JSON object from an LLM response.

    Returns ``success``, ``json_string``, ``error``, and ``cleaned_text``.
    """
    if not raw_text:
        return _failure("Empty response from LLM.")

    cleaned_text = _strip_markdown(raw_text)
    start_index = cleaned_text.find("{")
    if start_index == -1:
        return _failure("No JSON object found (missing '{').", cleaned_text)

    # Prefer the exact end of the first valid object, which avoids failing on
    # trailing commentary such as "Let me know if you need changes".
    candidate = cleaned_text[start_index:]
    json_string = _decode_prefix(candidate)
    if json_string is not None:
        return _success(json_string, cleaned_text)

    # Fallback: take everything between the first "{" and the last "}".
    end_index = cleaned_text.rfind("}")
    if end_index == -1 or start_index >= end_index:
        return _failure("No valid JSON object found.", cleaned_text)

    json_string = fix_trailing_commas(cleaned_text[start_index : end_index + 1])
    trimmed = _decode_prefix(json_string)
    if trimmed is not None:
        return _success(trimmed, cleaned_text)

    try:
        json.loads(json_string)
    except json.JSONDecodeError as error:
        return {
            "success": False,
            "json_string": json_string,
            "error": f"Extracted text is not valid JSON: {error}",
            "cleaned_text": cleaned_text,
        }
    return _success(json_string, cleaned_text)


def fix_trailing_commas(json_string: str) -> str:
    """Remove commas before closing braces/brackets and doubled commas."""
    fixed = json_string.replace(",}", "}").replace(",]", "]")
    return fixed.replace(",,", ",")


def parse_json_safely(json_string: str) -> Dict[str, Any]:
    """Parse JSON, tolerating trailing data after the first object."""
    try:
        data, _ = json.JSONDecoder().raw_decode(json_string)
        return {"success": True, "data": data, "error": None}
    except (json.JSONDecodeError, ValueError):
        try:
            return {"success": True, "data": json.loads(json_string), "error": None}
        except json.JSONDecodeError as error:
            return {"success": False, "data": None, "error": f"JSON parsing error: {error}"}


def _strip_markdown(raw_text: str) -> str:
    """Remove markdown fences and a leading byte-order mark."""
    text = raw_text.strip()
    for fence in ("```json", "```JSON", "```"):
        text = text.replace(fence, "")
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.strip()


def _decode_prefix(candidate: str) -> Optional[str]:
    """Return the first complete JSON object in ``candidate``, if any."""
    try:
        _, end_position = json.JSONDecoder().raw_decode(candidate)
    except (json.JSONDecodeError, ValueError):
        return None

    json_string = fix_trailing_commas(candidate[:end_position])
    try:
        json.loads(json_string)
    except json.JSONDecodeError:
        return None
    return json_string


def _success(json_string: str, cleaned_text: str) -> Dict[str, Any]:
    return {
        "success": True,
        "json_string": json_string,
        "error": None,
        "cleaned_text": cleaned_text,
    }


def _failure(error: str, cleaned_text: str = "") -> Dict[str, Any]:
    return {
        "success": False,
        "json_string": None,
        "error": error,
        "cleaned_text": cleaned_text,
    }
