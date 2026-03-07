# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Any, Callable, Dict, Union

from intergrax.llm_adapters.providers.aws_bedrock_adapter import BedrockChatAdapter
from intergrax.llm_adapters.providers.azure_openai_adapter import AzureOpenAIChatAdapter
from intergrax.llm_adapters.providers.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.providers.gemini_adapter import GeminiChatAdapter
from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.providers.mistral_adapter import MistralChatAdapter
from intergrax.llm_adapters.providers.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.providers.openai_responses_adapter import OpenAIChatResponsesAdapter

class LLMAdapterRegistry:
    _factories: Dict[str, Callable[..., LLMAdapter]] = {}

    @staticmethod
    def _normalize_provider(provider: Union[str, LLMProvider]) -> str:
        if isinstance(provider, LLMProvider):
            key = provider.value
        elif isinstance(provider, str):
            key = provider.strip()
        else:
            raise TypeError(f"provider must be str or LLMProvider, got {type(provider)!r}")

        if not key:
            raise ValueError("provider must not be empty")

        return key.lower()

    @classmethod
    def register(
        cls,
        provider: Union[str, LLMProvider],
        factory: Callable[..., LLMAdapter],
        *,
        override: bool = False,
    ) -> None:
        key = cls._normalize_provider(provider)
        if key in cls._factories and not override:
            raise ValueError(f"LLM adapter already registered for provider='{key}'")
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider: Union[str, LLMProvider], **kwargs) -> LLMAdapter:
        key = cls._normalize_provider(provider)

        if key not in cls._factories:
            raise ValueError(f"LLM adapter not registered for provider='{key}'")

        adapter = cls._factories[key](**kwargs)

        if not isinstance(adapter, LLMAdapter):
            raise TypeError(
                f"Factory for provider='{key}' returned invalid type "
                f"{type(adapter)!r}, expected LLMAdapter"
            )

        adapter.validate()
        return adapter


# Default adapter registrations
LLMAdapterRegistry.register(LLMProvider.OPENAI, lambda **kw: OpenAIChatResponsesAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.GEMINI, lambda **kw: GeminiChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.OLLAMA, lambda **kw: LangChainOllamaAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.CLAUDE, lambda **kw: ClaudeChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.MISTRAL, lambda **kw: MistralChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AZURE_OPENAI, lambda **kw: AzureOpenAIChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AWS_BEDROCK, lambda **kw: BedrockChatAdapter(**kw))