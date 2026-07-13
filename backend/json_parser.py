"""
=============================================================================
json_parser.py - Clean Raw LLM Output and Extract JSON
=============================================================================

What this module does:
    Takes the raw text response from an LLM and extracts clean JSON.
    
    LLMs often wrap JSON in extra text even when told not to:
    - Markdown code blocks: ```json ... ```
    - Explanations: "Here is the extracted data: {...}"
    - Trailing text: "..." + "Let me know if you need changes"
    - Stray characters: BOM marks, extra whitespace, trailing commas
    
    This module handles all of these cases so the rest of our code
    always receives clean, parseable JSON.

Why a separate module:
    In Phase 3, the JSON cleaning logic was inside extractor.py.
    As we add validation in Phase 4, the cleaning becomes more important.
    A separate module makes it:
    - Testable in isolation
    - Reusable across different extraction pipelines
    - Easy to improve without touching other code

=============================================================================
"""

# Import Python's JSON module for parsing
import json


def clean_llm_response(raw_text):
    if not raw_text:
        return {"success": False, "json_string": None, "error": "Empty response from LLM.", "cleaned_text": ""}

    cleaned_text = raw_text.strip()
    cleaned_text = cleaned_text.replace("```json", "").replace("```JSON", "").replace("```", "")

    if cleaned_text.startswith("\ufeff"):
        cleaned_text = cleaned_text[1:]

    cleaned_text = cleaned_text.strip()

    start_index = cleaned_text.find("{")
    if start_index == -1:
        return {"success": False, "json_string": None, "error": "No JSON object found (missing '{').", "cleaned_text": cleaned_text}

    # Use raw_decode to find the EXACT end of the first valid JSON object
    candidate = cleaned_text[start_index:]
    try:
        decoder = json.JSONDecoder()
        obj, end_pos = decoder.raw_decode(candidate)
        json_string = candidate[:end_pos]
        json_string = fix_trailing_commas(json_string)

        try:
            json.loads(json_string)
            return {"success": True, "json_string": json_string, "error": None, "cleaned_text": cleaned_text}
        except json.JSONDecodeError:
            pass
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract between first { and last }
    end_index = cleaned_text.rfind("}")
    if end_index == -1 or start_index >= end_index:
        return {"success": False, "json_string": None, "error": "No valid JSON object found.", "cleaned_text": cleaned_text}

    json_string = cleaned_text[start_index:end_index + 1]
    json_string = fix_trailing_commas(json_string)

    # Try full string first
    try:
        decoder = json.JSONDecoder()
        obj, end_pos = decoder.raw_decode(json_string)
        json_string = json_string[:end_pos]
        json.loads(json_string)
        return {"success": True, "json_string": json_string, "error": None, "cleaned_text": cleaned_text}
    except (json.JSONDecodeError, ValueError):
        pass

    # Final: try the whole extracted block
    try:
        json.loads(json_string)
        return {"success": True, "json_string": json_string, "error": None, "cleaned_text": cleaned_text}
    except json.JSONDecodeError as error:
        return {"success": False, "json_string": json_string, "error": f"Extracted text is not valid JSON: {error}", "cleaned_text": cleaned_text}


def fix_trailing_commas(json_string):
    """
    Remove trailing commas from JSON strings.
    
    What this does:
        Trailing commas are invalid in JSON but LLMs sometimes add them:
        {"a": 1, "b": 2,}  ← trailing comma before }
        
        This function removes commas that appear before } or ].
    
    Why this is needed:
        Python's json.loads() rejects trailing commas.
        Rather than crashing, we fix them before parsing.
    
    Parameters:
        json_string (str): The JSON string to fix
    
    Returns:
        str: The JSON string with trailing commas removed
    
    Python concepts used:
        - String replacement with careful ordering
        - Replacing patterns like ",}" with "}" and ",]" with "]"
    """
    
    # Remove trailing commas before closing braces
    # We need to do this carefully to avoid removing valid commas
    # Simple approach: replace ",}" with "}" and ",]" with "]"
    # We repeat because there might be multiple: {"a": 1, "b": 2,,}
    
    fixed = json_string
    
    # Remove trailing commas before closing braces
    fixed = fixed.replace(",}", "}")
    fixed = fixed.replace(",]", "]")
    
    # Handle double commas (rare but possible)
    fixed = fixed.replace(",,", ",")
    
    return fixed


def parse_json_safely(json_string):
    try:
        # Try raw_decode first to handle "Extra data" errors
        decoder = json.JSONDecoder()
        obj, end_pos = decoder.raw_decode(json_string)
        return {"success": True, "data": obj, "error": None}
    except (json.JSONDecodeError, ValueError):
        try:
            data = json.loads(json_string)
            return {"success": True, "data": data, "error": None}
        except json.JSONDecodeError as error:
            return {"success": False, "data": None, "error": f"JSON parsing error: {error}"}
