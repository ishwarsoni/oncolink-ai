"""
prompt_builder.py - Prompts for clinical data extraction.

Prompt wording is deliberately hand-written: small changes can alter what the
model extracts, so the text is not generated or deduplicated.

The field list below mirrors ``backend/schema.py`` (``ClinicalData``).  When a
field is added or removed there, update this prompt as well so the model is
asked for exactly the fields the validator accepts.
"""


def build_extraction_prompt(document_text: str) -> tuple[str, str]:
    """
    Return the ``(system_prompt, user_prompt)`` pair for extraction.

    The system prompt fixes the role and output format.  The user prompt lists
    the exact field names and types, forbids inference, and ends with a
    ``JSON:`` marker so the response starts where expected.
    """
    system_prompt = (
        "You are a clinical data extraction assistant specializing in oncology. "
        "Your task is to extract structured patient information from medical documents. "
        "Return ONLY valid JSON. No explanations, no markdown formatting, no extra text. "
        "Do not wrap the JSON in markdown code blocks."
    )

    user_prompt = f"""Extract the following clinical fields from the document(s) below.

Return a JSON object with these exact field names:

- "patient_name": The full name of the patient (string)
- "age": The patient's age in years (number)
- "gender": The patient's gender (string, e.g. "Male", "Female")
- "diagnosis": The complete diagnosis description (string)
- "cancer_type": The specific type of cancer (string, e.g. "Invasive Ductal Carcinoma")
- "cancer_stage": The cancer stage including TNM if available (string, e.g. "Stage IIB, T2 N1 M0")
- "biomarkers": A list of biomarkers found, each with "name", "value", and "status" (list of objects)
- "current_medication": Current medications the patient is taking (string)
- "previous_treatment": Any treatments the patient has already received (string)
- "ecog_score": The ECOG performance status score (number between 0-5, or null if not found)
- "follow_up_plan": The recommended follow-up schedule (string)
- "next_steps": The recommended next actions or plan (string)
- "adverse_events": Any side effects, toxicities, or adverse reactions to treatment reported (string, or null if none found)

Important rules:
- If a field is not found in the text, use null as the value
- Do NOT make up or infer information that is not explicitly stated
- Extract ONLY what is present in the provided text
- Return ONLY the JSON object, nothing else

Documents to analyze:
{ document_text }

JSON:
"""

    return system_prompt, user_prompt
