# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Factory for OpenAI Chat Completions-compatible LLM providers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


@dataclass(frozen=True)
class OpenAICompatProviderConfig:
    provider: LLMProvider
    api_key_env: str
    base_url_env: str
    default_base_url: str
    default_model: str
    context_windows: Dict[str, int]
    api_key_optional: bool = False
    missing_api_key_message: str = ""


def create_openai_compat_adapter(
    config: OpenAICompatProviderConfig,
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **defaults,
) -> OpenAIChatCompletionsAdapter:
    resolved_key = api_key or os.getenv(config.api_key_env)
    if client is None and not resolved_key and not config.api_key_optional:
        msg = config.missing_api_key_message or f"{config.api_key_env} not found in environment variables."
        raise RuntimeError(msg)

    resolved_model = model or config.default_model
    resolved_base = (base_url or os.getenv(config.base_url_env) or config.default_base_url).strip()

    if client is None:
        client = OpenAI(api_key=resolved_key or "EMPTY", base_url=resolved_base)

    return OpenAIChatCompletionsAdapter(
        client=client,
        model=resolved_model,
        provider=config.provider,
        context_windows=config.context_windows,
        **defaults,
    )
