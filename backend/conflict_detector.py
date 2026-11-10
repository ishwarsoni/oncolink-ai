"""
conflict_detector.py - Find fields where documents disagree.

Conflict detection runs on the *per-document* extractions, before
harmonization.  It reports a conflict whenever two or more documents contain
different non-null values for the same field, so the user can review the
source documents instead of trusting an arbitrary merged value.

Identity fields (name, age, gender, ECOG) are treated as high severity
because a disagreement there usually means the documents belong to different
patients or one extraction is wrong.
"""

import logging
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)

# Disagreement here is more serious than disagreement in a free-text field.
IDENTITY_FIELDS = ("patient_name", "age", "gender", "ecog_score")


def detect_conflicts(extraction_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Return ``success``, ``conflicts``, and ``documents_compared``."""
    results = list(extraction_results or [])
    if not results:
        return {"success": True, "conflicts": [], "documents_compared": 0}

    valid_results = [
        result for result in results if result.get("success") and result.get("data")
    ]
    if len(valid_results) < 2:
        return {"success": True, "conflicts": [], "documents_compared": len(valid_results)}

    field_entries = _group_values_by_field(valid_results)
    conflicts: List[Dict[str, Any]] = []

    for field_name, entries in field_entries.items():
        non_null = [entry for entry in entries if entry["value"] is not None]
        if len(non_null) < 2:
            continue

        unique_values = {_comparable_value(entry["value"]) for entry in non_null}
        if len(unique_values) <= 1:
            continue

        if field_name == "biomarkers":
            conflicts.extend(_detect_biomarker_conflicts(non_null))
            continue

        conflicts.append(
            {
                "field": field_name,
                "severity": "high" if field_name in IDENTITY_FIELDS else "medium",
                "sources": [(entry["filename"], entry["value"]) for entry in non_null],
                "description": (
                    f"Conflict in '{field_name}': "
                    f"{len(unique_values)} different values across documents"
                ),
            }
        )

    logger.info("Conflict detection complete: %d conflict(s) found", len(conflicts))
    return {
        "success": True,
        "conflicts": conflicts,
        "documents_compared": len(valid_results),
    }


def _group_values_by_field(
    valid_results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Map each field name to its value in every document."""
    field_entries: Dict[str, List[Dict[str, Any]]] = {}
    for result in valid_results:
        filename = result.get("filename", "Unknown")
        for field_name, value in result["data"].items():
            field_entries.setdefault(field_name, []).append(
                {"filename": filename, "value": value}
            )
    return field_entries


def _comparable_value(value: Any) -> Any:
    """
    Convert a value into something that can be compared across documents.

    Strings are compared case-insensitively and without surrounding
    whitespace.  Biomarker lists compare as sets of (name, value) pairs so a
    different ordering is not reported as a conflict.
    """
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return tuple(
            (item.get("name", "").lower(), item.get("value", "").lower())
            for item in value
            if isinstance(item, dict)
        )
    return str(value)


def _detect_biomarker_conflicts(non_null_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Report one conflict per biomarker whose value differs across documents."""
    values_by_name: Dict[str, Dict[str, Any]] = {}

    for entry in non_null_entries:
        biomarkers = entry["value"]
        if not isinstance(biomarkers, list):
            continue
        for biomarker in biomarkers:
            if not isinstance(biomarker, dict):
                continue

            name = biomarker.get("name", "").strip().lower()
            if not name:
                continue

            record = values_by_name.setdefault(
                name, {"canonical": biomarker.get("name", ""), "values": {}}
            )
            record["values"][entry["filename"]] = biomarker.get("value", "").strip().lower()

    conflicts = []
    for info in values_by_name.values():
        document_values = info["values"]
        if len(set(document_values.values())) <= 1:
            continue
        conflicts.append(
            {
                "field": f"biomarker.{info['canonical']}",
                "severity": "high",
                "sources": list(document_values.items()),
                "description": (
                    f"Biomarker '{info['canonical']}' has conflicting values "
                    "across documents"
                ),
            }
        )
    return conflicts
