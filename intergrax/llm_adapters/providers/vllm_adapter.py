# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

import os
from typing import Dict, Optional

from openai import OpenAI

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


class VllmChatAdapter(OpenAIChatCompletionsAdapter):
    """
    vLLM OpenAI-compatible server adapter (local or cluster inference).

    Point at your vLLM ``--api-key`` and ``/v1`` base URL.
    """

    DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
    DEFAULT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

    ENV_API_KEY = "VLLM_API_KEY"
    ENV_MODEL = "INTERGRAX_DEFAULT_VLLM_MODEL"
    ENV_BASE_URL = "INTERGRAX_DEFAULT_VLLM_BASE_URL"

    _VLLM_CONTEXT_WINDOWS: Dict[str, int] = {
        "meta-llama/Llama-3.1-8B-Instruct": 128_000,
        "meta-llama/Llama-3.1-70B-Instruct": 128_000,
    }

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **defaults,
    ):
        resolved_base = (base_url or os.getenv(self.ENV_BASE_URL) or self.DEFAULT_BASE_URL).strip()
        resolved_model = model or os.getenv(self.ENV_MODEL) or self.DEFAULT_MODEL
        resolved_key = api_key or os.getenv(self.ENV_API_KEY) or "EMPTY"

        client = client or OpenAI(api_key=resolved_key, base_url=resolved_base)
        super().__init__(
            client=client,
            model=resolved_model,
            provider=LLMProvider.VLLM,
            context_windows=self._VLLM_CONTEXT_WINDOWS,
            **defaults,
        )
