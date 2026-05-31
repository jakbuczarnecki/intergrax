# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""OpenAI Chat Completions-compatible provider adapters."""

from __future__ import annotations

import os
from typing import Optional

from openai import OpenAI

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.openai_compat_factory import (
    OpenAICompatProviderConfig,
    create_openai_compat_adapter,
)
from intergrax.llm_adapters.providers.openai_chat_completions_adapter import OpenAIChatCompletionsAdapter


class _CompatAdapterBase(OpenAIChatCompletionsAdapter):
    _CONFIG: OpenAICompatProviderConfig

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **defaults,
    ) -> None:
        built = create_openai_compat_adapter(
            self._CONFIG,
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            **defaults,
        )
        self.__dict__.update(built.__dict__)


class GroqChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.GROQ,
        api_key_env="GROQ_API_KEY",
        model_env="INTERGRAX_DEFAULT_GROQ_MODEL",
        base_url_env="INTERGRAX_DEFAULT_GROQ_BASE_URL",
        default_base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        context_windows={"llama-3.3-70b-versatile": 128_000, "llama-3.1-8b-instant": 128_000},
    )


class VllmChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.VLLM,
        api_key_env="VLLM_API_KEY",
        model_env="INTERGRAX_DEFAULT_VLLM_MODEL",
        base_url_env="INTERGRAX_DEFAULT_VLLM_BASE_URL",
        default_base_url="http://127.0.0.1:8000/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct",
        context_windows={"meta-llama/Llama-3.1-8B-Instruct": 128_000},
        api_key_optional=True,
    )


class TogetherChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.TOGETHER,
        api_key_env="TOGETHER_API_KEY",
        model_env="INTERGRAX_DEFAULT_TOGETHER_MODEL",
        base_url_env="INTERGRAX_DEFAULT_TOGETHER_BASE_URL",
        default_base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.1-8B-Instruct-Turbo",
        context_windows={},
    )


class FireworksChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.FIREWORKS,
        api_key_env="FIREWORKS_API_KEY",
        model_env="INTERGRAX_DEFAULT_FIREWORKS_MODEL",
        base_url_env="INTERGRAX_DEFAULT_FIREWORKS_BASE_URL",
        default_base_url="https://api.fireworks.ai/inference/v1",
        default_model="accounts/fireworks/models/llama-v3p1-8b-instruct",
        context_windows={},
    )


class OpenRouterChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.OPENROUTER,
        api_key_env="OPENROUTER_API_KEY",
        model_env="INTERGRAX_DEFAULT_OPENROUTER_MODEL",
        base_url_env="INTERGRAX_DEFAULT_OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        context_windows={},
    )


class DeepSeekChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.DEEPSEEK,
        api_key_env="DEEPSEEK_API_KEY",
        model_env="INTERGRAX_DEFAULT_DEEPSEEK_MODEL",
        base_url_env="INTERGRAX_DEFAULT_DEEPSEEK_BASE_URL",
        default_base_url="https://api.deepseek.com",
        default_model="deepseek-chat",
        context_windows={"deepseek-chat": 64_000, "deepseek-reasoner": 64_000},
    )


class XaiChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.XAI,
        api_key_env="XAI_API_KEY",
        model_env="INTERGRAX_DEFAULT_XAI_MODEL",
        base_url_env="INTERGRAX_DEFAULT_XAI_BASE_URL",
        default_base_url="https://api.x.ai/v1",
        default_model="grok-2-latest",
        context_windows={"grok-2-latest": 131_072},
    )


class LlamaCppChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.LLAMA_CPP,
        api_key_env="LLAMA_CPP_API_KEY",
        model_env="INTERGRAX_DEFAULT_LLAMA_CPP_MODEL",
        base_url_env="INTERGRAX_DEFAULT_LLAMA_CPP_BASE_URL",
        default_base_url="http://127.0.0.1:8080/v1",
        default_model="default",
        context_windows={},
        api_key_optional=True,
    )


class CohereChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.COHERE,
        api_key_env="COHERE_API_KEY",
        model_env="INTERGRAX_DEFAULT_COHERE_MODEL",
        base_url_env="INTERGRAX_DEFAULT_COHERE_BASE_URL",
        default_base_url="https://api.cohere.com/compatibility/v1",
        default_model="command-r-plus",
        context_windows={"command-r-plus": 128_000},
    )


class AzureAiInferenceChatAdapter(_CompatAdapterBase):
    _CONFIG = OpenAICompatProviderConfig(
        provider=LLMProvider.AZURE_AI_INFERENCE,
        api_key_env="AZURE_AI_INFERENCE_API_KEY",
        model_env="INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_MODEL",
        base_url_env="INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_BASE_URL",
        default_base_url="",
        default_model="gpt-4o-mini",
        context_windows={},
        missing_api_key_message=(
            "AZURE_AI_INFERENCE_API_KEY and INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_BASE_URL are required."
        ),
    )

    def __init__(
        self,
        client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **defaults,
    ) -> None:
        resolved_base = (
            base_url or os.getenv(self._CONFIG.base_url_env) or self._CONFIG.default_base_url
        ).strip()
        if not resolved_base and client is None:
            raise RuntimeError(
                "INTERGRAX_DEFAULT_AZURE_AI_INFERENCE_BASE_URL must be set "
                "(e.g. https://<resource>.services.ai.azure.com/models/openai/v1)."
            )
        super().__init__(
            client=client,
            model=model,
            base_url=resolved_base,
            api_key=api_key,
            **defaults,
        )
