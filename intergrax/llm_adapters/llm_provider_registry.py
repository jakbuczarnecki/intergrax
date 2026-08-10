# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations
import importlib
from typing import Any, Callable, Dict, Tuple, Union, cast

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class LLMAdapterDependencyError(RuntimeError):
    """Raised when a selected LLM provider SDK is not installed."""


class LLMAdapterRegistrationError(RuntimeError):
    """Raised when a builtin LLM provider registration is invalid."""


_BUILTIN_ADAPTERS: Dict[str, Tuple[str, str]] = {
    LLMProvider.OPENAI.value: (
        "intergrax.llm_adapters.providers.openai_responses_adapter",
        "OpenAIChatResponsesAdapter",
    ),
    LLMProvider.GEMINI.value: (
        "intergrax.llm_adapters.providers.gemini_adapter",
        "GeminiChatAdapter",
    ),
    LLMProvider.VERTEX_GEMINI.value: (
        "intergrax.llm_adapters.providers.vertex_gemini_adapter",
        "VertexGeminiChatAdapter",
    ),
    LLMProvider.OLLAMA.value: (
        "intergrax.llm_adapters.providers.native_ollama_adapter",
        "NativeOllamaAdapter",
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
    LLMProvider.GROQ.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "GroqChatAdapter",
    ),
    LLMProvider.VLLM.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "VllmChatAdapter",
    ),
    LLMProvider.TOGETHER.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "TogetherChatAdapter",
    ),
    LLMProvider.FIREWORKS.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "FireworksChatAdapter",
    ),
    LLMProvider.OPENROUTER.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "OpenRouterChatAdapter",
    ),
    LLMProvider.DEEPSEEK.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "DeepSeekChatAdapter",
    ),
    LLMProvider.XAI.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "XaiChatAdapter",
    ),
    LLMProvider.LLAMA_CPP.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "LlamaCppChatAdapter",
    ),
    LLMProvider.COHERE.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "CohereChatAdapter",
    ),
    LLMProvider.COHERE_NATIVE.value: (
        "intergrax.llm_adapters.providers.cohere_native_adapter",
        "CohereNativeChatAdapter",
    ),
    LLMProvider.AZURE_AI_INFERENCE.value: (
        "intergrax.llm_adapters.providers.openai_compat_providers",
        "AzureAiInferenceChatAdapter",
    ),
}

_BUILTIN_OPTIONAL_DEPENDENCIES: Dict[str, Tuple[Tuple[str, ...], str, str]] = {
    LLMProvider.OPENAI.value: (("openai",), "openai", "llm-openai"),
    LLMProvider.GEMINI.value: (("google", "google.genai"), "google-genai", "llm-gemini"),
    LLMProvider.VERTEX_GEMINI.value: (
        ("google", "google.genai"),
        "google-genai",
        "llm-vertex",
    ),
    LLMProvider.OLLAMA.value: (("ollama",), "ollama", "llm-ollama"),
    LLMProvider.CLAUDE.value: (("anthropic",), "anthropic", "llm-anthropic"),
    LLMProvider.MISTRAL.value: (("mistralai",), "mistralai", "llm-mistral"),
    LLMProvider.AZURE_OPENAI.value: (("openai",), "openai", "llm-openai"),
    LLMProvider.AWS_BEDROCK.value: (
        ("boto3", "mypy_boto3_bedrock_runtime"),
        "boto3",
        "llm-bedrock",
    ),
    LLMProvider.GROQ.value: (("openai",), "openai", "llm-groq"),
    LLMProvider.VLLM.value: (("openai",), "openai", "llm-vllm"),
    LLMProvider.TOGETHER.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.FIREWORKS.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.OPENROUTER.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.DEEPSEEK.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.XAI.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.LLAMA_CPP.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.COHERE.value: (("openai",), "openai", "llm-compat"),
    LLMProvider.COHERE_NATIVE.value: (("cohere",), "cohere", "llm-cohere-native"),
    LLMProvider.AZURE_AI_INFERENCE.value: (("openai",), "openai", "llm-compat"),
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
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            dependency = _BUILTIN_OPTIONAL_DEPENDENCIES.get(key)
            if dependency is None or exc.name not in dependency[0]:
                raise
            _, dependency_name, extra_name = dependency
            raise LLMAdapterDependencyError(
                f"LLM provider '{key}' requires dependency '{dependency_name}'. "
                f"Install it with 'Intergrax-ai[{extra_name}]' before selecting "
                "this provider."
            ) from exc
        except ImportError:
            raise

        try:
            adapter_cls = cast(Callable[..., LLMAdapter], getattr(module, class_name))
        except AttributeError as exc:
            raise LLMAdapterRegistrationError(
                f"LLM provider '{key}' module '{module_path}' does not define "
                f"expected adapter class '{class_name}'."
            ) from exc

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

        from intergrax.llm_adapters.registry.catalog_capabilities import (
            enrich_adapter_with_catalog_capabilities,
        )

        from intergrax.utils import attribute_access

        model_id = kwargs.get("model") or attribute_access.optional(adapter, "model", None)
        adapter = enrich_adapter_with_catalog_capabilities(
            adapter,
            provider=key,
            model=str(model_id) if model_id else None,
        )

        adapter.validate()
        return adapter

    @classmethod
    def registered_providers(cls) -> list[str]:
        keys = set(cls._factories.keys()) | set(_BUILTIN_ADAPTERS.keys())
        return sorted(keys)
