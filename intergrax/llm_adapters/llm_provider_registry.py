# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Tuple, Union

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

# Lazy factory: (module path, class name)
_BUILTIN_ADAPTERS: Dict[str, Tuple[str, str]] = {
    LLMProvider.OPENAI.value: (
        "intergrax.llm_adapters.providers.openai_responses_adapter",
        "OpenAIChatResponsesAdapter",
    ),
    LLMProvider.GEMINI.value: (
        "intergrax.llm_adapters.providers.gemini_adapter",
        "GeminiChatAdapter",
    ),
    LLMProvider.OLLAMA.value: (
        "intergrax.llm_adapters.providers.ollama_adapter",
        "LangChainOllamaAdapter",
    ),
    LLMProvider.CLAUDE.value: (
        "intergrax.llm_adapters.providers.claude_adapter",
        "ClaudeChatAdapter",
    ),
    LLMProvider.MISTRAL.value: (
        "intergrax.llm_adapters.providers.mistral_adapter",
        "MistralChatAdapter",
    ),
    LLMProvider.AZURE_OPENAI.value: (
        "intergrax.llm_adapters.providers.azure_openai_adapter",
        "AzureOpenAIChatAdapter",
    ),
    LLMProvider.AWS_BEDROCK.value: (
        "intergrax.llm_adapters.providers.aws_bedrock_adapter",
        "BedrockChatAdapter",
    ),
}


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
    def _ensure_builtin(cls, key: str) -> None:
        if key in cls._factories:
            return
        spec = _BUILTIN_ADAPTERS.get(key)
        if spec is None:
            return
        module_path, class_name = spec
        module = importlib.import_module(module_path)
        adapter_cls = getattr(module, class_name)

        def factory(**kwargs: Any) -> LLMAdapter:
            return adapter_cls(**kwargs)

        cls._factories[key] = factory

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
        cls._ensure_builtin(key)

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

    @classmethod
    def registered_providers(cls) -> list[str]:
        """Return provider keys with a registered or built-in factory."""
        keys = set(cls._factories.keys()) | set(_BUILTIN_ADAPTERS.keys())
        return sorted(keys)
