"""Centralised NVIDIA NIM client used by extraction and RAG generation.

NVIDIA NIM exposes an OpenAI-compatible API, so the standard ``openai`` client
works once ``base_url`` points at NVIDIA's endpoint.  This module is the only
place in the codebase that constructs an LLM client or reads the API key,
which keeps model configuration, timeouts and secret handling consistent.

Secret handling:
    The key is read from the ``NVIDIA_API_KEY`` environment variable.  The UI
    layer resolves Streamlit secrets and the sidebar field into that variable
    before any pipeline runs, so this module never imports Streamlit and
    never writes the key to logs.

Model configuration:
    ``DEFAULT_MODEL`` (Nemotron 3 Ultra) and ``NVIDIA_BASE_URL`` are preserved
    from the production configuration.  Override the model only through
    ``ONCOLINK_RAG_MODEL`` / an explicit argument.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

# A single request can legitimately take a while for long clinical documents.
# The explicit timeout prevents a stalled connection from hanging a Streamlit
# session forever; retries handle transient transport failures.
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 2

# Value shipped in the example secrets file; it must never be treated as a key.
_PLACEHOLDER_API_KEY = "nvapi-your-key-here"


class LLMConfigurationError(ValueError):
    """Raised when no usable NVIDIA API key is configured."""


def get_api_key() -> str:
    """Return the NVIDIA API key from the environment.

    Raises:
        LLMConfigurationError: when no usable key is configured.
    """
    api_key = (os.getenv("NVIDIA_API_KEY") or "").strip()
    if not api_key or api_key == _PLACEHOLDER_API_KEY:
        raise LLMConfigurationError(
            "NVIDIA_API_KEY not found.\n\n"
            "To fix this:\n"
            "  - **Local**: set NVIDIA_API_KEY in `.env`\n"
            "  - **Streamlit Cloud**: set the secret `NVIDIA_API_KEY`\n"
            "  - **Anywhere**: enter the key in the sidebar\n\n"
            "Get a free key at: https://build.nvidia.com/"
        )
    return api_key


def create_nvidia_client(timeout: Optional[float] = None) -> OpenAI:
    """Create an OpenAI-compatible client pointed at NVIDIA NIM.

    A new HTTP client is created per pipeline operation rather than per LLM
    call, so connection pooling is reused across a batch of documents.
    """
    return OpenAI(
        base_url=NVIDIA_BASE_URL,
        api_key=get_api_key(),
        timeout=float(timeout or DEFAULT_TIMEOUT_SECONDS),
        max_retries=DEFAULT_MAX_RETRIES,
    )


def call_llm(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    top_p: float = 0.1,
    max_tokens: int = 4096,
    reasoning_budget: Optional[int] = None,
    enable_thinking: bool = False,
) -> str:
    """Send a streaming chat-completion request and return the full text.

    Transport errors are returned as a string beginning with
    ``"LLM API call failed"`` so callers can surface a readable message
    without exception plumbing.  This contract is relied on by the pipelines
    and their tests.
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
        logger.error("NVIDIA LLM call failed: %s", error)
        return f"LLM API call failed: {error}"
