"""
=============================================================================
prompt_builder.py - Prompt Templates for Clinical Data Extraction
=============================================================================

What this module does:
    Creates the prompts (instructions) we send to the LLM.
    Contains carefully designed prompt templates that tell the model
    exactly what to extract and how to format its response.

Why Prompt Engineering Matters:
    LLMs are incredibly flexible — they can respond in many ways to the
    same input. If we just say "extract data", the model might:
    - Return data in a random format
    - Include explanations we don't want
    - Miss important fields
    - Hallucinate (make up) information not in the text
    
    A well-designed prompt prevents all of these by being:
    1. SPECIFIC: Tell the model exactly what fields to extract
    2. STRUCTURED: Request JSON output with field names
    3. CONSTRAINED: Set rules (no markdown, no explanations)
    4. EXAMPLES: Show the expected output format

Why JSON Output:
    JSON (JavaScript Object Notation) is a standard way to structure data.
    It's easy for both computers and humans to read.
    
    Example JSON:
    {
        "patient_name": "Meera Sharma",
        "age": 58,
        "diagnosis": "Invasive Ductal Carcinoma"
    }
    
    We prefer JSON because:
    - Python can parse it with json.loads() into a dictionary
    - It's easy to display in a UI
    - It can be saved to files or databases
    - It's the standard for API data exchange

What is a System Prompt vs User Prompt:
    - SYSTEM PROMPT: Sets the model's behavior and persona.
      "You are a clinical extraction assistant..."
      This is like giving someone a job description before they start work.
    
    - USER PROMPT: The specific task input.
      "Extract data from this document: [text]"
      This is the actual work assignment.
    
    The model processes both together, but the system prompt has more
    weight in guiding behavior.

How to Reduce Hallucinations:
    - Tell the model: "If a field is not found, use null"
    - Set low temperature (0.1) to reduce creativity
    - Be explicit about the input source
    - Never ask the model to "guess" missing information

=============================================================================
"""


def build_extraction_prompt(document_text):
    """
    Build system and user prompts for clinical data extraction.
    
    This function creates a pair of prompts (system + user) that instruct
    the LLM to extract structured clinical data from oncology documents.
    
    Parameters:
        document_text (str): The raw text extracted from uploaded documents.
            This could be from a single document or combined from multiple.
            We add document separators to help the model understand context.
    
    How the prompt is structured:
        1. Role definition (system prompt): "You are a clinical extraction
           assistant" — sets the model's expertise and behavior
        2. Format instruction: "Return ONLY valid JSON" — prevents markdown
        3. Field definitions: Lists each field with its expected data type
        4. Hallucination guard: "If a field is not found, use null"
        5. Input section: The actual document text, clearly separated
        6. Output marker: "JSON:" — tells the model where to start its output
    
    Returns:
        tuple: (system_prompt, user_prompt) — both are strings ready to
               send to the LLM
    
    Python concepts used:
        - Docstring (triple-quoted string for function documentation)
        - f-strings for embedding variables in strings
        - Returning multiple values as a tuple
    """
    
    # =========================================================================
    # System Prompt: Sets the model's behavior and role
    # =========================================================================
    # This tells the model WHO it is and HOW it should behave.
    # It stays consistent across all extraction tasks.
    system_prompt = (
        "You are a clinical data extraction assistant specializing in oncology. "
        "Your task is to extract structured patient information from medical documents. "
        "Return ONLY valid JSON. No explanations, no markdown formatting, no extra text. "
        "Do not wrap the JSON in markdown code blocks."
    )
    
    # =========================================================================
    # User Prompt: The specific extraction task
    # =========================================================================
    # This contains the actual document text and the list of fields to extract.
    # We use an f-string to embed the document_text variable.
    #
    # The prompt design principles used:
    # 1. Clear instructions up front ("Extract the following clinical fields")
    # 2. Exact field names the JSON should use (snake_case for Python compatibility)
    # 3. Data types for each field (string, number, list, null)
    # 4. Anti-hallucination instruction ("use null as the value")
    # 5. Clear separation between instructions and input data
    # 6. Output marker ("JSON:") to anchor the model's response
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
