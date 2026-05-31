# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import os
from typing import Dict, Optional

from openai import OpenAI

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


class GroqChatAdapter(OpenAIChatCompletionsAdapter):
    """
    Groq adapter (OpenAI-compatible Chat Completions API).

    https://console.groq.com/docs/openai
    """

    DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    ENV_API_KEY = "GROQ_API_KEY"
    ENV_MODEL = "INTERGRAX_DEFAULT_GROQ_MODEL"
    ENV_BASE_URL = "INTERGRAX_DEFAULT_GROQ_BASE_URL"

    _GROQ_CONTEXT_WINDOWS: Dict[str, int] = {
        "llama-3.3-70b-versatile": 128_000,
        "llama-3.1-8b-instant": 128_000,
        "mixtral-8x7b-32768": 32_768,
        "gemma2-9b-it": 8_192,
    }

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        **defaults,
    ):
        api_key = os.getenv(self.ENV_API_KEY)
        if client is None and not api_key:
            raise RuntimeError("GROQ_API_KEY not found in environment variables.")

        resolved_model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL
        resolved_base = (base_url or os.getenv(self.ENV_BASE_URL) or self.DEFAULT_BASE_URL).strip()

        client = client or OpenAI(api_key=api_key, base_url=resolved_base)
        super().__init__(
            client=client,
            model=resolved_model,
            provider=LLMProvider.GROQ,
            context_windows=self._GROQ_CONTEXT_WINDOWS,
            **defaults,
        )
