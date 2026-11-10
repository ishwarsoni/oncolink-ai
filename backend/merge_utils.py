"""
merge_utils.py - Small, single-purpose helpers for merging field values.

The harmonizer merges different kinds of fields in different ways, so each
strategy lives in its own function:

    first_non_null          identity/single values (patient_name, age, ecog)
    prefer_longest_string   free text where detail matters (diagnosis)
    merge_unique_strings    semicolon-separated lists (medications, plans)
    merge_biomarker_lists   structured biomarker entries
"""

import logging
from typing import Any, Iterable, List, Optional

logger = logging.getLogger(__name__)


def first_non_null(values: Iterable[Any]) -> Optional[Any]:
    """Return the first value that is not ``None``."""
    for value in values:
        if value is not None:
            return value
    return None


def first_non_empty_string(strings: Iterable[Any]) -> Optional[str]:
    """Return the first string that contains more than whitespace."""
    for text in strings:
        if text and isinstance(text, str) and text.strip():
            return text.strip()
    return None


def prefer_longest_string(strings: Iterable[Any]) -> Optional[str]:
    """Return the longest non-empty string, since it usually has more detail."""
    valid_strings = [
        text.strip()
        for text in strings
        if text and isinstance(text, str) and text.strip()
    ]
    return max(valid_strings, key=len) if valid_strings else None


def merge_unique_strings(strings: Iterable[Any]) -> Optional[str]:
    """
    Combine semicolon/comma/newline separated values into one deduplicated
    string.

    Near-duplicates are collapsed by keeping the longer of two strings that
    contain each other (for example "AC-T" and "AC-T chemotherapy").  This
    prevents the same treatment appearing twice with slightly different
    wording from different documents.
    """
    seen = set()
    merged_items: List[str] = []

    for text in strings:
        if not text or not isinstance(text, str):
            continue

        for part in text.replace(";", ",").replace("\n", ",").split(","):
            cleaned = part.strip()
            if not cleaned or cleaned.lower() in seen:
                continue

            duplicate = _find_contained_item(merged_items, cleaned)
            if duplicate is None:
                seen.add(cleaned.lower())
                merged_items.append(cleaned)
                continue

            # Keep the longer, more descriptive wording.
            if len(cleaned) > len(duplicate):
                merged_items.remove(duplicate)
                seen.discard(duplicate.lower())
                merged_items.append(cleaned)
                seen.add(cleaned.lower())

    return "; ".join(merged_items) if merged_items else None


def _find_contained_item(existing_items: List[str], candidate: str) -> Optional[str]:
    """Return an existing item that overlaps the candidate, if any."""
    candidate_lower = candidate.lower()
    if len(candidate_lower) < 10:
        return None

    for existing in existing_items:
        existing_lower = existing.lower()
        if len(existing_lower) >= 10 and (
            candidate_lower in existing_lower or existing_lower in candidate_lower
        ):
            return existing
    return None


def merge_biomarker_lists(biomarker_lists: Iterable[Any]) -> List[dict]:
    """
    Merge biomarker lists, keeping the first entry for each name.

    Different values for the same biomarker are intentionally not merged
    here: conflict detection needs to see that the documents disagreed.
    """
    seen_names = set()
    merged: List[dict] = []

    for biomarker_list in biomarker_lists:
        if not isinstance(biomarker_list, list):
            continue

        for biomarker in biomarker_list:
            if not isinstance(biomarker, dict):
                continue

            name = biomarker.get("name", "").strip().lower()
            if not name:
                continue
            if name in seen_names:
                logger.info("Duplicate biomarker %r found in multiple documents", biomarker.get("name"))
                continue

            seen_names.add(name)
            merged.append(biomarker)

    return merged
