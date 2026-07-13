import logging

logger = logging.getLogger(__name__)


def generate_patient_summary(harmonized_data, field_sources=None, conflicts=None):
    if not harmonized_data or not isinstance(harmonized_data, dict):
        return {"success": False, "summary": None, "error": "No harmonized data available."}

    try:
        lines = []
        lines.append("=" * 60)
        lines.append("PATIENT SUMMARY")
        lines.append("=" * 60)
        lines.append("")

        name = harmonized_data.get("patient_name", "Unknown")
        age = harmonized_data.get("age", "Unknown")
        gender = harmonized_data.get("gender", "Unknown")
        lines.append(f"Patient: {name} | Age: {age} | Gender: {gender}")
        lines.append("")

        if conflicts:
            conflict_list = [c for c in conflicts if c.get("severity") == "high"]
            if conflict_list:
                lines.append("!" * 60)
                lines.append("CONFLICTS DETECTED")
                lines.append("!" * 60)
                for c in conflict_list:
                    field_label = c.get("field", "Unknown field").replace("_", " ").title()
                    lines.append(f"  ! {field_label}: {c['description']}")
                    for doc, val in c.get("sources", []):
                        lines.append(f"    {doc}: {val}")
                lines.append("")

        lines.append("-" * 60)
        lines.append("DIAGNOSIS & STAGING")
        lines.append("-" * 60)
        diagnosis = harmonized_data.get("diagnosis")
        if diagnosis:
            lines.append(f"  Diagnosis: {diagnosis}")
        cancer_type = harmonized_data.get("cancer_type")
        if cancer_type:
            lines.append(f"  Cancer Type: {cancer_type}")
        stage = harmonized_data.get("cancer_stage")
        if stage:
            lines.append(f"  Stage: {stage}")
        ecog = harmonized_data.get("ecog_score")
        if ecog is not None:
            lines.append(f"  ECOG Performance Status: {ecog}")
        lines.append("")

        biomarkers = harmonized_data.get("biomarkers", [])
        if biomarkers:
            lines.append("-" * 60)
            lines.append("BIOMARKERS")
            lines.append("-" * 60)
            for bm in biomarkers:
                name_bm = bm.get("name", "Unknown")
                value = bm.get("value", "N/A")
                status = bm.get("status")
                if status:
                    lines.append(f"  {name_bm}: {value} ({status})")
                else:
                    lines.append(f"  {name_bm}: {value}")
            lines.append("")

        meds = harmonized_data.get("current_medication")
        if meds:
            lines.append("-" * 60)
            lines.append("CURRENT MEDICATIONS")
            lines.append("-" * 60)
            for item in meds.split(";"):
                item = item.strip()
                if item:
                    lines.append(f"  - {item}")
            lines.append("")

        prev_tx = harmonized_data.get("previous_treatment")
        if prev_tx:
            lines.append("-" * 60)
            lines.append("PREVIOUS TREATMENT")
            lines.append("-" * 60)
            for item in prev_tx.split(";"):
                item = item.strip()
                if item:
                    lines.append(f"  - {item}")
            lines.append("")

        follow_up = harmonized_data.get("follow_up_plan")
        if follow_up:
            lines.append("-" * 60)
            lines.append("FOLLOW-UP PLAN")
            lines.append("-" * 60)
            for item in follow_up.split(";"):
                item = item.strip()
                if item:
                    lines.append(f"  - {item}")
            lines.append("")

        next_steps = harmonized_data.get("next_steps")
        if next_steps:
            lines.append("-" * 60)
            lines.append("NEXT STEPS")
            lines.append("-" * 60)
            for item in next_steps.split(";"):
                item = item.strip()
                if item:
                    lines.append(f"  - {item}")
            lines.append("")

        adverse = harmonized_data.get("adverse_events")
        if adverse:
            lines.append("-" * 60)
            lines.append("ADVERSE EVENTS")
            lines.append("-" * 60)
            for item in adverse.split(";"):
                item = item.strip()
                if item:
                    lines.append(f"  - {item}")
            lines.append("")

        lines.append("=" * 60)
        summary_text = "\n".join(lines)

        logger.info("Patient summary generated successfully")

        return {
            "success": True,
            "summary": summary_text,
            "error": None
        }

    except Exception as error:
        logger.error(f"Summary generation failed: {error}")
        return {
            "success": False,
            "summary": None,
            "error": f"Summary generation failed: {error}"
        }
