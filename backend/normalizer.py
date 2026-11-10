"""
normalizer.py - Standardize clinical values after validation.

Normalization makes equivalent values comparable, which matters because the
harmonizer and conflict detector compare values as strings:

    "Stage ii"  -> "Stage II"
    "IDC"       -> "Invasive Ductal Carcinoma"
    "F"         -> "Female"

Normalization never invents missing values and never changes a medical fact;
it only cleans, expands known abbreviations, and fixes capitalization.
"""

import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# Abbreviation maps.  Keys are lowercase so matching is case-insensitive.
CANCER_TYPE_MAP = {
    "idc": "Invasive Ductal Carcinoma",
    "ilc": "Invasive Lobular Carcinoma",
    "dcis": "Ductal Carcinoma In Situ",
    "lcis": "Lobular Carcinoma In Situ",
    "tnbc": "Triple Negative Breast Cancer",
    "mbc": "Metastatic Breast Cancer",
    "nsclc": "Non-Small Cell Lung Cancer",
    "sclc": "Small Cell Lung Cancer",
    "hcc": "Hepatocellular Carcinoma",
    "ccrcc": "Clear Cell Renal Cell Carcinoma",
}

GENDER_MAP = {
    "m": "Male",
    "f": "Female",
    "male": "Male",
    "female": "Female",
}

# Compound names collapse to a canonical gene symbol.  The variant detail is
# preserved in the biomarker's "value" field instead of the name.
BIOMARKER_NAME_MAP = {
    "er": "ER",
    "pr": "PR",
    "her2": "HER2",
    "her-2": "HER2",
    "ki67": "Ki-67",
    "ki-67": "Ki-67",
    "egfr": "EGFR",
    "egfr exon 19 deletion": "EGFR",
    "egfr exon19 deletion": "EGFR",
    "egfr mutation": "EGFR",
    "egfr mutant": "EGFR",
    "egfr positive": "EGFR",
    "egfr negative": "EGFR",
    "egfr wild type": "EGFR",
    "egfr wildtype": "EGFR",
    "alk": "ALK",
    "alk rearrangement": "ALK",
    "alk fusion": "ALK",
    "alk positive": "ALK",
    "alk negative": "ALK",
    "ros1": "ROS1",
    "ros1 rearrangement": "ROS1",
    "ros1 fusion": "ROS1",
    "ros1 positive": "ROS1",
    "ros1 negative": "ROS1",
    "pd-l1": "PD-L1",
    "pdl1": "PD-L1",
    "pd l1": "PD-L1",
}

BIOMARKER_GENE_PREFIXES = {
    "egfr": "EGFR",
    "alk": "ALK",
    "ros1": "ROS1",
    "pd-l1": "PD-L1",
    "pdl1": "PD-L1",
    "her2": "HER2",
    "braf": "BRAF",
    "kras": "KRAS",
    "nras": "NRAS",
    "pik3ca": "PIK3CA",
}

BIOMARKER_STATUS_MAP = {
    "+": "Positive",
    "pos": "Positive",
    "positive": "Positive",
    "-": "Negative",
    "neg": "Negative",
    "negative": "Negative",
    "equivocal": "Equivocal",
    "borderline": "Borderline",
}

# Values that mean "present" rather than carrying a measurement.  A gene
# suffix replaces these because the suffix is more informative.
_GENERIC_BIOMARKER_VALUES = {"", "detected", "positive", "negative", "+", "-"}

_TEXT_FIELDS = (
    "diagnosis",
    "current_medication",
    "previous_treatment",
    "follow_up_plan",
    "next_steps",
)


def normalize_clinical_data(data_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a normalized copy of validated clinical data.

    The input is never modified so callers can still inspect the original
    values for debugging or audit purposes.
    """
    logger.info("Normalization started")
    normalized = dict(data_dict)

    changes = 0
    changes += _normalize_patient_name(normalized, data_dict)
    changes += _normalize_gender(normalized, data_dict)
    changes += _normalize_cancer_type(normalized, data_dict)
    changes += _normalize_cancer_stage(normalized, data_dict)
    changes += _normalize_ecog_score(normalized, data_dict)
    changes += _normalize_biomarkers(normalized)
    _trim_text_fields(normalized)

    logger.info("Normalization complete: %d normalizations performed", changes)
    return normalized


def _normalize_patient_name(normalized: Dict[str, Any], original: Dict[str, Any]) -> int:
    """Title-case the patient name so "meera sharma" becomes "Meera Sharma"."""
    name = normalized.get("patient_name")
    if not (name and isinstance(name, str)):
        return 0
    cleaned = name.strip().title()
    normalized["patient_name"] = cleaned
    logger.debug("Normalized patient_name: %r -> %r", original.get("patient_name"), cleaned)
    return 1


def _normalize_gender(normalized: Dict[str, Any], original: Dict[str, Any]) -> int:
    """Expand gender abbreviations (M/F) to their full form."""
    gender = normalized.get("gender")
    if not (gender and isinstance(gender, str)):
        return 0

    key = gender.strip().lower()
    normalized["gender"] = GENDER_MAP.get(key, gender.strip().capitalize())
    logger.debug("Normalized gender: %r -> %r", original.get("gender"), normalized["gender"])
    return 1


def _normalize_cancer_type(normalized: Dict[str, Any], original: Dict[str, Any]) -> int:
    """Expand known cancer-type abbreviations."""
    cancer_type = normalized.get("cancer_type")
    if not (cancer_type and isinstance(cancer_type, str)):
        return 0

    key = cancer_type.strip().lower()
    normalized["cancer_type"] = CANCER_TYPE_MAP.get(key, cancer_type.strip())
    logger.debug(
        "Normalized cancer_type: %r -> %r",
        original.get("cancer_type"),
        normalized["cancer_type"],
    )
    return 1


def _normalize_cancer_stage(normalized: Dict[str, Any], original: Dict[str, Any]) -> int:
    """Capitalize the word "stage" while preserving the TNM detail."""
    stage = normalized.get("cancer_stage")
    if not (stage and isinstance(stage, str)):
        return 0

    stage = stage.strip()
    if stage.lower().startswith("stage"):
        stage = "Stage" + stage[5:]
    normalized["cancer_stage"] = stage
    logger.debug(
        "Normalized cancer_stage: %r -> %r", original.get("cancer_stage"), stage
    )
    return 1


def _normalize_ecog_score(normalized: Dict[str, Any], original: Dict[str, Any]) -> int:
    """Convert values such as "ECOG 1" or 1.0 to the integer 1."""
    ecog = normalized.get("ecog_score")
    if ecog is None:
        return 0

    if isinstance(ecog, str):
        numbers = re.findall(r"\d+", ecog)
        if not numbers:
            return 0
        normalized["ecog_score"] = int(numbers[0])
        logger.debug(
            "Normalized ecog_score: %r -> %r",
            original.get("ecog_score"),
            normalized["ecog_score"],
        )
        return 1

    if isinstance(ecog, float):
        normalized["ecog_score"] = int(ecog)
        return 1
    return 0


def _normalize_biomarkers(normalized: Dict[str, Any]) -> int:
    """
    Normalize each biomarker and merge duplicate names in one document.

    Two entries with the same canonical name are merged rather than dropped so
    a more specific value (e.g. "exon 19 deletion") is not lost.
    """
    biomarkers = normalized.get("biomarkers", [])
    if not (biomarkers and isinstance(biomarkers, list)):
        return 0

    result: List[Dict[str, Any]] = []
    index_by_name: Dict[str, int] = {}

    for biomarker in biomarkers:
        normalized_biomarker = normalize_single_biomarker(biomarker)
        name_key = normalized_biomarker.get("name", "").strip().lower()

        if name_key and name_key in index_by_name:
            _merge_duplicate_biomarker(
                result[index_by_name[name_key]], normalized_biomarker, name_key
            )
            continue

        if name_key:
            index_by_name[name_key] = len(result)
        result.append(normalized_biomarker)

    normalized["biomarkers"] = result
    return 0


def _merge_duplicate_biomarker(
    existing: Dict[str, Any], candidate: Dict[str, Any], name_key: str
) -> None:
    """Keep the longer value when the same biomarker appears twice."""
    existing_value = existing.get("value", "").strip()
    candidate_value = candidate.get("value", "").strip()
    if (
        candidate_value
        and candidate_value.lower() not in existing_value.lower()
        and len(candidate_value) > len(existing_value)
    ):
        existing["value"] = candidate_value
    logger.debug("Merged duplicate biomarker %r in same document", name_key)


def _trim_text_fields(normalized: Dict[str, Any]) -> None:
    """Strip surrounding whitespace from free-text fields."""
    for field_name in _TEXT_FIELDS:
        value = normalized.get(field_name)
        if value and isinstance(value, str):
            normalized[field_name] = value.strip()


def normalize_single_biomarker(biomarker: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize one biomarker's name, value, and status."""
    normalized = dict(biomarker)

    name = normalized.get("name", "")
    if name and isinstance(name, str):
        normalized["name"] = _normalize_biomarker_name(normalized, name.strip())

    value = normalized.get("value", "")
    if value and isinstance(value, str):
        value_key = value.strip().lower()
        if value_key in BIOMARKER_STATUS_MAP:
            normalized["value"] = BIOMARKER_STATUS_MAP[value_key]

    status = normalized.get("status")
    if status and isinstance(status, str):
        status_key = status.strip().lower()
        if status_key in BIOMARKER_STATUS_MAP:
            normalized["status"] = BIOMARKER_STATUS_MAP[status_key]

    return normalized


def _normalize_biomarker_name(normalized: Dict[str, Any], name: str) -> str:
    """
    Resolve a biomarker name to its canonical gene symbol.

    A name such as "EGFR exon 19 deletion" becomes name="EGFR" with the
    variant detail moved into the value when the value is generic.
    """
    name_lower = name.lower()

    if name_lower in BIOMARKER_NAME_MAP:
        return BIOMARKER_NAME_MAP[name_lower]

    for prefix, canonical in BIOMARKER_GENE_PREFIXES.items():
        if not (name_lower.startswith(prefix) and len(name) > len(prefix)):
            continue

        suffix = name[len(prefix):].strip().lstrip(" ,-–")
        existing_value = normalized.get("value", "").strip()
        if suffix and existing_value.lower() in _GENERIC_BIOMARKER_VALUES:
            if existing_value.lower() not in ("", "detected"):
                normalized["value"] = f"{suffix} ({existing_value})"
            else:
                normalized["value"] = suffix
        return canonical

    return name


def run_normalization_pipeline(data_dict: Any) -> Dict[str, Any]:
    """Run normalization and return a success/error envelope."""
    try:
        if not isinstance(data_dict, dict):
            return {
                "success": False,
                "data": None,
                "error": "Cannot normalize: input is not a dictionary.",
            }

        return {
            "success": True,
            "data": normalize_clinical_data(data_dict),
            "error": None,
        }
    except Exception as error:  # defensive: normalization is non-critical
        logger.error("Normalization pipeline failed: %s", error)
        return {
            "success": False,
            "data": None,
            "error": f"Normalization failed: {error}",
        }
