"""
llm_client.py - NVIDIA NIM chat-completion client.

NVIDIA NIM exposes an OpenAI-compatible API, so the standard ``openai`` client
works once ``base_url`` points at NVIDIA's endpoint.  Extraction uses a low
temperature and a large token budget because the task is precision, not
creativity.

The API key is never stored in this file: it is read from Streamlit secrets,
the environment, or the sidebar (which exports it to the environment).
"""

import os
from typing import Optional

from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# Value shipped in .env.example; it must never be treated as a real key.
_PLACEHOLDER_API_KEY = "nvapi-your-key-here"


def get_api_key() -> str:
    """
    Return the NVIDIA API key from Streamlit secrets or the environment.

    Raises:
        ValueError: when no usable key is configured.
    """
    api_key = _key_from_streamlit_secrets() or os.getenv("NVIDIA_API_KEY")

    if not api_key or api_key == _PLACEHOLDER_API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY not found.\n\n"
            "To fix this:\n"
            "  - **Local**: Open the `.env` file and replace 'nvapi-your-key-here' with your actual key\n"
            "  - **Streamlit Cloud**: In your app dashboard, set the secret `NVIDIA_API_KEY`\n"
            "  - **Anywhere**: Enter your key in the sidebar input field\n\n"
            "Get a free key at: https://build.nvidia.com/"
        )

    return api_key


def _key_from_streamlit_secrets():
    """Read the key from Streamlit secrets when they are available."""
    try:
        import streamlit as st

        if "NVIDIA_API_KEY" in st.secrets:
            return st.secrets["NVIDIA_API_KEY"]
    except Exception:
        # Secrets are optional outside Streamlit Cloud.
        return None
    return None


def create_nvidia_client() -> OpenAI:
    """Create an OpenAI-compatible client pointed at NVIDIA NIM."""
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=get_api_key())


def call_llm(
    client,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    top_p: float = 0.1,
    max_tokens: int = 4096,
    reasoning_budget: Optional[int] = None,
    enable_thinking: bool = False,
) -> str:
    """
    Send a chat-completion request and return the response text.

    Transport errors are returned as a string beginning with
    ``"LLM API call failed"`` so callers can handle them without exceptions.
    """
    try:
        request_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": True,
        }

        if enable_thinking and reasoning_budget:
            request_params["extra_body"] = {
                "chat_template_kwargs": {"enable_thinking": True},
                "reasoning_budget": reasoning_budget,
            }

        full_content = ""
        for chunk in client.chat.completions.create(**request_params):
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content is not None:
                full_content += content
        return full_content

    except Exception as error:  # defensive: callers expect a string result
        return f"LLM API call failed: {error}"
