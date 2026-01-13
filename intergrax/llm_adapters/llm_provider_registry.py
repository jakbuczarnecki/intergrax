# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from typing import Any, Callable, Dict, Union

from intergrax.llm_adapters.aws_bedrock_adapter import BedrockChatAdapter
from intergrax.llm_adapters.azure_openai_adapter import AzureOpenAIChatAdapter
from intergrax.llm_adapters.claude_adapter import ClaudeChatAdapter
from intergrax.llm_adapters.gemini_adapter import GeminiChatAdapter
from intergrax.llm_adapters.llm_adapter import LLMAdapter
from intergrax.llm_adapters.llm_provider import LLMProvider
from intergrax.llm_adapters.mistral_adapter import MistralChatAdapter
from intergrax.llm_adapters.ollama_adapter import LangChainOllamaAdapter
from intergrax.llm_adapters.openai_responses_adapter import OpenAIChatResponsesAdapter

class LLMAdapterRegistry:
    _factories: Dict[str, Any] = {}

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
    def register(cls, provider: Union[str, LLMProvider], factory: Callable[..., LLMAdapter]) -> None:
        key = cls._normalize_provider(provider)
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider: Union[str, LLMProvider], **kwargs) -> LLMAdapter:
        key = cls._normalize_provider(provider)
        if key not in cls._factories:
            raise ValueError(f"LLM adapter not registered for provider='{key}'")
        return cls._factories[key](**kwargs)


# Default adapter registrations
LLMAdapterRegistry.register(LLMProvider.OPENAI, lambda **kw: OpenAIChatResponsesAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.GEMINI, lambda **kw: GeminiChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.OLLAMA, lambda **kw: LangChainOllamaAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.CLAUDE, lambda **kw: ClaudeChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.MISTRAL, lambda **kw: MistralChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AZURE_OPENAI, lambda **kw: AzureOpenAIChatAdapter(**kw))
LLMAdapterRegistry.register(LLMProvider.AWS_BEDROCK, lambda **kw: BedrockChatAdapter(**kw))