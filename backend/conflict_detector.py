import logging

logger = logging.getLogger(__name__)


def detect_conflicts(extraction_results):
    if not extraction_results:
        return {"success": True, "conflicts": [], "documents_compared": 0}

    valid_results = [r for r in extraction_results if r.get("success") and r.get("data")]
    if len(valid_results) < 2:
        return {"success": True, "conflicts": [], "documents_compared": len(valid_results)}

    conflicts = []
    field_values = {}

    for result in valid_results:
        data = result["data"]
        filename = result.get("filename", "Unknown")
        for field, value in data.items():
            if field not in field_values:
                field_values[field] = []
            field_values[field].append({"filename": filename, "value": value})

    identity_fields = ["patient_name", "age", "gender", "ecog_score"]

    for field, entries in field_values.items():
        non_null = [e for e in entries if e["value"] is not None]
        if len(non_null) < 2:
            continue

        unique_values = set()
        for e in non_null:
            v = e["value"]
            if isinstance(v, str):
                unique_values.add(v.strip().lower())
            elif isinstance(v, (int, float)):
                unique_values.add(str(v))
            elif isinstance(v, list):
                unique_values.add(tuple(
                    (bm.get("name", "").lower(), bm.get("value", "").lower())
                    for bm in v if isinstance(bm, dict)
                ))
            else:
                unique_values.add(str(v))

        if len(unique_values) > 1:
            if field == "biomarkers":
                bm_conflicts = _detect_biomarker_conflicts(non_null)
                conflicts.extend(bm_conflicts)
            else:
                severity = "high" if field in identity_fields else "medium"
                sources = [(e["filename"], e["value"]) for e in non_null]
                conflicts.append({
                    "field": field,
                    "severity": severity,
                    "sources": sources,
                    "description": f"Conflict in '{field}': {len(unique_values)} different values across documents"
                })

    logger.info(f"Conflict detection complete: {len(conflicts)} conflict(s) found")

    return {
        "success": True,
        "conflicts": conflicts,
        "documents_compared": len(valid_results)
    }


def _detect_biomarker_conflicts(non_null_entries):
    conflicts = []
    bm_map = {}

    for entry in non_null_entries:
        filename = entry["filename"]
        biomarkers = entry["value"]
        if not isinstance(biomarkers, list):
            continue
        for bm in biomarkers:
            if not isinstance(bm, dict):
                continue
            name = bm.get("name", "").strip().lower()
            if not name:
                continue
            value = bm.get("value", "").strip().lower()
            if name not in bm_map:
                bm_map[name] = {}
                bm_map[name]["canonical"] = bm.get("name", "")
            bm_map[name][filename] = value

    for bm_name_lower, info in bm_map.items():
        doc_values = {k: v for k, v in info.items() if k != "canonical"}
        unique_vals = set(doc_values.values())
        if len(unique_vals) > 1:
            sources = [(doc, val) for doc, val in doc_values.items()]
            conflicts.append({
                "field": f"biomarker.{info['canonical']}",
                "severity": "high",
                "sources": sources,
                "description": f"Biomarker '{info['canonical']}' has conflicting values across documents"
            })

    return conflicts
