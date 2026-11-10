"""Compatibility import path for the NVIDIA LLM client.

The canonical implementation lives in :mod:`infrastructure.llm`.
"""

from infrastructure.llm import (
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    NVIDIA_BASE_URL,
    LLMConfigurationError,
    call_llm,
    create_nvidia_client,
    get_api_key,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_SECONDS",
    "NVIDIA_BASE_URL",
    "LLMConfigurationError",
    "call_llm",
    "create_nvidia_client",
    "get_api_key",
]
