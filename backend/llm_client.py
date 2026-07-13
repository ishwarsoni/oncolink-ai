"""
=============================================================================
llm_client.py - NVIDIA NIM API Client
=============================================================================

What this module does:
    Provides functions to connect to NVIDIA's NIM (NVIDIA Inference 
    Microservice) API and send chat completion requests.

    NVIDIA NIM provides an OpenAI-compatible API. This means we can use
    the standard "openai" Python library with a custom base URL pointing
    to NVIDIA's servers instead of OpenAI's servers.

Key Concepts:

    1. LLM API:
       An API (Application Programming Interface) lets our Python code
       send a request to a remote server running an AI model. We send
       text (our prompt), the model processes it, and sends text back.
       
       Think of it like ordering food: we send our order (prompt) to
       the restaurant (API server), the chef (LLM) prepares the food,
       and the waiter brings it back (response).
    
    2. Token:
       LLMs don't read words — they read "tokens". A token is roughly
       3/4 of a word. "Hello world" is ~2 tokens. "I am a doctor" is ~4.
       Models have a "context window" (max tokens they can process at once).
    
    3. Temperature:
       Controls how "creative" the model is. Range: 0.0 to 2.0.
       - 0.0: Always picks the most likely next word (deterministic)
       - 0.5: Some variation but mostly focused
       - 1.0: Balanced creativity
       - 2.0: Very random, may produce nonsense
       
       For structured data extraction, use LOW temperature (0.0-0.2).
       We want the model to be precise and consistent, not creative.
    
    4. max_tokens:
       The maximum number of tokens the model can generate in its response.
       This limits response length and cost. For extraction, 2000 tokens
       is usually enough.

Why NVIDIA NIM instead of OpenAI directly:
    - NVIDIA provides free credits for their API catalog
    - OpenAI-compatible (same code works for both)
    - Models like Llama 3.1 8B are competitive with GPT-3.5 for extraction

=============================================================================
"""

# Import the OpenAI client library
# OpenAI v2.0+ uses this pattern: from openai import OpenAI
# The client can connect to ANY OpenAI-compatible API, not just OpenAI's.
from openai import OpenAI

# Import os for reading environment variables
import os


def get_api_key():
    """
    Get the NVIDIA API key from environment variables or Streamlit secrets.
    
    How this works:
        1. Checks Streamlit secrets first (for Streamlit Cloud deployments)
        2. Falls back to os.getenv (for local .env files or sidebar input)
        3. If the key isn't set, we raise an error with helpful instructions.
    
    Why environment variables:
        - API keys are SECRETS. If hardcoded in code, they get uploaded
          to GitHub and anyone can use them (costing you money).
        - .env file is listed in .gitignore, so Git ignores it.
        - Other developers can clone the project and use their own .env.
    
    Returns:
        str: The NVIDIA API key
        
    Raises:
        ValueError: If the API key is not found in environment variables
    """
    api_key = None
    
    # Priority 1: Streamlit secrets (for Streamlit Cloud deployments)
    try:
        import streamlit as st
        if "NVIDIA_API_KEY" in st.secrets:
            api_key = st.secrets["NVIDIA_API_KEY"]
    except Exception:
        pass
    
    # Priority 2: Environment variables (local .env or sidebar input)
    if not api_key:
        api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key or api_key == "nvapi-your-key-here":
        raise ValueError(
            "NVIDIA_API_KEY not found.\n\n"
            "To fix this:\n"
            "  - **Local**: Open the `.env` file and replace 'nvapi-your-key-here' with your actual key\n"
            "  - **Streamlit Cloud**: In your app dashboard, set the secret `NVIDIA_API_KEY`\n"
            "  - **Anywhere**: Enter your key in the sidebar input field\n\n"
            "Get a free key at: https://build.nvidia.com/"
        )
    
    return api_key


def create_nvidia_client():
    """
    Create an OpenAI-compatible client configured for NVIDIA NIM.
    
    How this works:
        The OpenAI client library normally connects to OpenAI's servers
        at https://api.openai.com. By changing the "base_url", we redirect
        it to NVIDIA's servers instead.
        
        NVIDIA's API follows the same format as OpenAI's, so all the same
        function calls work — we just change the address.
    
    Returns:
        OpenAI: A configured client object ready to make API calls
    
    Python concepts used:
        - Creating an object from a class (OpenAI())
        - The object stores configuration (base_url, api_key) for later use
    """
    
    # Get the API key from environment variables
    api_key = get_api_key()
    
    # Create the client with NVIDIA's base URL
    # base_url is the server address NVIDIA's API lives at
    # api_key authenticates us to use the API
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=api_key
    )
    
    return client


def call_llm(
    client,
    system_prompt,
    user_prompt,
    model="nvidia/nemotron-3-ultra-550b-a55b",
    temperature=1.0,
    top_p=0.95,
    max_tokens=16384,
    reasoning_budget=16384,
    enable_thinking=True
):
    """
    Send a chat completion request to the LLM and get the response.
    
    Parameters:
        client (OpenAI): The NVIDIA client object (created above)
        system_prompt (str): System-level instructions for the model
        user_prompt (str): The actual input/question for the model
        model (str): NVIDIA model to use
        temperature (float): 0.0-2.0, creativity control
        top_p (float): Nucleus sampling threshold
        max_tokens (int): Maximum response tokens
        reasoning_budget (int): Max tokens for reasoning/thinking
        enable_thinking (bool): Enable chain-of-thought reasoning
    
    The "messages" list structure:
        A chat conversation is a list of messages, each with a "role":
        - "system": Sets the model's behavior (system prompt)
        - "user": The human's input (our extraction request)
        - "assistant": The model's response (what we're generating)
        
        We send system + user, and the model generates the assistant response.
    
    Returns:
        str: The text content of the model's response (thinking + content combined)
    """
    
    try:
        # Build request params
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }
        
        if enable_thinking:
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": reasoning_budget
            }
        
        # Stream the response and collect chunks
        stream = client.chat.completions.create(**request_params)
        
        full_content = ""
        full_reasoning = ""
        
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                full_reasoning += reasoning
            
            if delta.content is not None:
                full_content += delta.content
        
        # Combine reasoning + content
        if full_reasoning and full_content:
            response_text = full_reasoning + "\n" + full_content
        else:
            response_text = full_content or full_reasoning
        
        return response_text
    
    except Exception as error:
        error_message = f"LLM API call failed: {error}"
        return error_message
