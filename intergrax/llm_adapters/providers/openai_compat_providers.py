# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""OpenAI Chat Completions-compatible provider adapters."""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

from openai import OpenAI

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.adapter_response import LLMAdapterResponse
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.contracts.structured_result import LLMStructuredResult
from intergrax.llm_adapters.contracts.stream_event import LLMStreamEvent
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
        **defaults: Any,
    ) -> None:
        self._delegate = create_openai_compat_adapter(
            self._CONFIG,
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            **defaults,
        )
        self.provider = self._delegate.provider
        self.model = self._delegate.model
        self.call_config = self._delegate.call_config
        self.usage = self._delegate.usage

    def generate_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return self._delegate.generate_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def stream_messages(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        return self._delegate.stream_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def generate_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> LLMAdapterResponse:
        return self._delegate.generate_with_tools(
            messages,
            tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    def stream_with_tools(
        self,
        messages: Sequence[ChatMessage],
        tools_schema: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        run_id: Optional[str] = None,
    ) -> Iterable[LLMStreamEvent]:
        return self._delegate.stream_with_tools(
            messages,
            tools_schema,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
            run_id=run_id,
        )

    def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        output_model: type,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> LLMStructuredResult[Any]:
        return self._delegate.generate_structured(
            messages,
            output_model,
            temperature=temperature,
            max_tokens=max_tokens,
            run_id=run_id,
        )

    def supports_streaming(self) -> bool:
        return self._delegate.supports_streaming()

    def supports_tools(self) -> bool:
        return self._delegate.supports_tools()

    def supports_structured_output(self) -> bool:
        return self._delegate.supports_structured_output()

    def supports_vision(self) -> bool:
        return self._delegate.supports_vision()

    def supports_audio_input(self) -> bool:
        return self._delegate.supports_audio_input()

    def supports_audio_output(self) -> bool:
        return self._delegate.supports_audio_output()

    @property
    def context_window_tokens(self) -> int:
        return self._delegate.context_window_tokens

    def validate(self) -> None:
        self._delegate.validate()


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
        **defaults: Any,
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
