"""LLM client for OpenRouter-compatible chat completion."""

from __future__ import annotations

import os
from typing import Any, Dict, List

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    OpenAI = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class LLMClient:
    """Minimal OpenRouter LLM client using an OpenAI-compatible SDK."""

    def __init__(
        self,
        model: str = "nex-agi/deepseek-v3.1-nex-n1:free",
        base_url: str = "https://openrouter.ai/api/v1",
        api_key_env: str = "OPENROUTER_API_KEY",
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            model: OpenRouter model identifier.
            base_url: OpenRouter API base URL.
            api_key_env: Environment variable name storing the API key.
        """
        if OpenAI is None:
            raise ImportError(
                "openai SDK is required. Please install it via requirements.txt."
            ) from _IMPORT_ERROR

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"Missing API key. Please set environment variable {api_key_env}."
            )

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self._model = model

    def generate(self, messages: List[Dict[str, Any]]) -> str:
        """
        Generate a response using chat completion.

        Args:
            messages: List of chat messages following OpenAI format.

        Returns:
            Generated assistant response as a string.
        """
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            return f"Error: {exc}"
