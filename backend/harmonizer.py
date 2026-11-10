"""
harmonizer.py - Merge per-document extractions into one patient record.

Each uploaded document produces its own extraction.  Harmonization combines
those extractions into a single record using a strategy chosen per field:

    identity fields (name, age, gender)   first non-null value
    text fields (diagnosis, stage)        longest, most detailed value
    biomarkers                            union of unique names
    treatment/plan fields                 union of unique semicolon items
    single values (ECOG)                  first non-null value

The strategy for every field is explicit so reviewers can see why a value
was chosen.  ``field_sources`` records which document supplied each value,
which the summary and conflict views rely on.
"""

import logging
from typing import Any, Callable, Dict, Iterable, List, Tuple

from backend.merge_utils import (
    first_non_null,
    merge_biomarker_lists,
    merge_unique_strings,
    prefer_longest_string,
)

logger = logging.getLogger(__name__)

IDENTITY_FIELDS = ("patient_name", "age", "gender")
TEXT_FIELDS = ("diagnosis", "cancer_type", "cancer_stage")
SINGLE_VALUE_FIELDS = ("ecog_score",)
STRING_FIELDS = ("current_medication", "previous_treatment", "adverse_events")
PLAN_FIELDS = ("follow_up_plan", "next_steps")


def harmonize_extractions(extraction_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge multiple extraction results into one unified patient record.

    Returns ``success``, ``data``, ``documents_merged``, ``field_sources``
    and ``warnings``.  Failed extractions are skipped and logged.
    """
    logger.info("Starting harmonization")
    valid_data_list, document_names = _collect_valid_extractions(extraction_results)

    if not valid_data_list:
        logger.error("No valid extraction data to harmonize")
        return {
            "success": False,
            "data": None,
            "documents_merged": 0,
            "field_sources": {},
            "warnings": ["No valid extraction data available to harmonize."],
        }

    logger.info("Harmonizing %d document(s)", len(valid_data_list))
    unified, field_sources = _merge_all_fields(valid_data_list, document_names)

    logger.info("Harmonization complete: %d documents merged", len(valid_data_list))
    logger.info("Unified record has %d fields", len(unified))
    return {
        "success": True,
        "data": unified,
        "documents_merged": len(valid_data_list),
        "field_sources": field_sources,
        "warnings": [],
    }


def _collect_valid_extractions(
    extraction_results: Iterable[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Split successful extractions from failed ones, preserving order."""
    valid_data_list: List[Dict[str, Any]] = []
    document_names: List[str] = []

    for result in extraction_results or []:
        document_name = result.get("filename", "Unknown document")
        if result.get("success") and result.get("data") is not None:
            valid_data_list.append(result["data"])
            document_names.append(document_name)
            logger.info("Adding data from document: %s", document_name)
        else:
            logger.warning("Skipping failed extraction for: %s", document_name)

    return valid_data_list, document_names


def _merge_all_fields(
    valid_data_list: List[Dict[str, Any]], document_names: List[str]
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Apply each field strategy and return the record plus its sources."""
    unified: Dict[str, Any] = {}
    field_sources: Dict[str, str] = {}

    merge_identity(unified, field_sources, valid_data_list, document_names, IDENTITY_FIELDS)
    merge_text(unified, field_sources, valid_data_list, document_names, TEXT_FIELDS)
    merge_biomarkers(unified, field_sources, valid_data_list, document_names)
    merge_string_fields(unified, field_sources, valid_data_list, document_names)
    merge_single_values(
        unified, field_sources, valid_data_list, document_names, SINGLE_VALUE_FIELDS
    )
    merge_plan_fields(unified, field_sources, valid_data_list, document_names, PLAN_FIELDS)

    return unified, field_sources


# ---------------------------------------------------------------------------
# Merge strategies
# ---------------------------------------------------------------------------


def merge_identity(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
    fields: Iterable[str],
) -> None:
    """Take the first non-null value for fields that should agree."""
    for field_name in fields:
        values = _collect_values(valid_data_list, field_name)
        merged = first_non_null(values)
        if merged is None:
            continue

        unified[field_name] = merged
        _record_first_source(
            field_sources, field_name, values, document_names, lambda value: value is not None
        )
        logger.debug("Identity merge %r from %r", field_name, field_sources.get(field_name))


def merge_text(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
    fields: Iterable[str],
) -> None:
    """Prefer the longest value, which is usually the most detailed."""
    for field_name in fields:
        values = _collect_values(valid_data_list, field_name)
        merged = prefer_longest_string(values)
        if merged is None:
            continue

        unified[field_name] = merged
        _record_first_source(
            field_sources,
            field_name,
            values,
            document_names,
            lambda value: isinstance(value, str) and value.strip() == merged,
        )
        logger.debug("Text merge %r from %r", field_name, field_sources.get(field_name))


def merge_biomarkers(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
) -> None:
    """Combine biomarker entries from all documents, deduplicated by name."""
    biomarker_lists = [
        data["biomarkers"]
        for data in valid_data_list
        if isinstance(data.get("biomarkers"), list) and data["biomarkers"]
    ]

    if not biomarker_lists:
        unified["biomarkers"] = []
        return

    merged = merge_biomarker_lists(biomarker_lists)
    unified["biomarkers"] = merged

    source_documents = [
        document_names[index]
        for index, data in enumerate(valid_data_list)
        if index < len(document_names)
        and isinstance(data.get("biomarkers"), list)
        and data["biomarkers"]
    ]
    if source_documents:
        field_sources["biomarkers"] = "; ".join(source_documents)

    logger.debug(
        "Biomarker merge: %d unique biomarkers from %d document(s)",
        len(merged),
        len(source_documents),
    )


def merge_string_fields(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
) -> None:
    """Combine semicolon-separated treatment and adverse-event lists."""
    _merge_unique_string_fields(
        unified, field_sources, valid_data_list, document_names, STRING_FIELDS
    )


def merge_plan_fields(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
    fields: Iterable[str],
) -> None:
    """Combine plan items and resolve conflicting "start"/"continue" advice."""
    _merge_unique_string_fields(
        unified, field_sources, valid_data_list, document_names, fields
    )


def merge_single_values(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
    fields: Iterable[str],
) -> None:
    """Take the first non-null numeric value (ECOG score)."""
    for field_name in fields:
        values = _collect_values(valid_data_list, field_name)
        merged = first_non_null(values)
        if merged is None:
            continue

        unified[field_name] = merged
        _record_first_source(
            field_sources, field_name, values, document_names, lambda value: value is not None
        )
        logger.debug("Single value merge %r: %r", field_name, merged)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _collect_values(
    valid_data_list: List[Dict[str, Any]], field_name: str
) -> List[Any]:
    """Return one field's value from every document, in document order."""
    return [data.get(field_name) for data in valid_data_list]


def _record_first_source(
    field_sources: Dict[str, str],
    field_name: str,
    values: List[Any],
    document_names: List[str],
    matches: Callable[[Any], bool],
) -> None:
    """Record the first document whose value satisfies ``matches``."""
    for index, value in enumerate(values):
        if index < len(document_names) and matches(value):
            field_sources[field_name] = document_names[index]
            return


def _record_all_sources(
    field_sources: Dict[str, str],
    field_name: str,
    values: List[Any],
    document_names: List[str],
    matches: Callable[[Any], bool],
) -> None:
    """Record every document that contributed a value."""
    sources = [
        document_names[index]
        for index, value in enumerate(values)
        if index < len(document_names) and matches(value)
    ]
    if sources:
        field_sources[field_name] = "; ".join(sources)


def _merge_unique_string_fields(
    unified: Dict[str, Any],
    field_sources: Dict[str, str],
    valid_data_list: List[Dict[str, Any]],
    document_names: List[str],
    fields: Iterable[str],
) -> None:
    """Shared implementation for treatment and plan string fields."""
    for field_name in fields:
        values = _collect_values(valid_data_list, field_name)
        merged = merge_unique_strings(values)
        if merged is None:
            continue

        if field_name == "next_steps":
            merged = _resolve_next_steps(merged)

        unified[field_name] = merged
        _record_all_sources(
            field_sources,
            field_name,
            values,
            document_names,
            lambda value: isinstance(value, str) and bool(value.strip()),
        )
        logger.debug("String merge %r: %d characters", field_name, len(merged))


def _resolve_next_steps(merged: str) -> str:
    """
    Drop "start ..." items when another document says to continue treatment.

    A document that says "continue current therapy" is more authoritative than
    one that says "start ..." for the same plan, so the conflicting start
    instruction is removed rather than presented as a parallel action.
    """
    items = [item.strip() for item in merged.split(";")]
    if any(item.lower().startswith("continue") for item in items if item):
        items = [item for item in items if not (item and item.lower().startswith("start"))]
    return "; ".join(items) if items else merged
