"""Compatibility import path for the clinical domain schema.

The canonical implementation lives in :mod:`domain.clinical`.  This module is
kept so existing tests and integrations that import ``backend.schema`` continue
to work during the architecture migration.
"""

from domain.clinical import (
    CLINICAL_SCHEMA,
    FIELD_LABELS,
    REQUIRED_FIELDS,
    BiomarkerItem,
    ClinicalData,
    create_validation_result,
    get_field_label,
    get_required_fields,
)

__all__ = [
    "CLINICAL_SCHEMA",
    "FIELD_LABELS",
    "REQUIRED_FIELDS",
    "BiomarkerItem",
    "ClinicalData",
    "create_validation_result",
    "get_field_label",
    "get_required_fields",
]
