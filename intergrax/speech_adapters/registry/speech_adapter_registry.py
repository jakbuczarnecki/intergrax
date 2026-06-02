# © Artur Czarnecki. All rights reserved.

"""Factory registry for ``SpeechAdapter`` implementations."""

from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter
from intergrax.speech_adapters.contracts.speech_provider import SpeechProvider

_BUILTIN_SPEECH_ADAPTERS: dict[str, tuple[str, str]] = {
    SpeechProvider.STUB.value: (
        "intergrax.speech_adapters.providers.stub_speech",
        "StubSpeechAdapter",
    ),
    SpeechProvider.ELEVENLABS.value: (
        "intergrax.speech_adapters.providers.elevenlabs_speech",
        "ElevenLabsSpeechAdapter",
    ),
}


class SpeechAdapterRegistry:
    """Create speech adapters by provider slug (same pattern as ``LLMAdapterRegistry``)."""

    _factories: dict[str, Callable[..., SpeechAdapter]] = {}

    @staticmethod
    def _normalize_provider(provider: str | SpeechProvider) -> str:
        if isinstance(provider, SpeechProvider):
            return provider.value
        if isinstance(provider, str) and provider.strip():
            return provider.strip().lower()
        raise ValueError("provider must be a non-empty SpeechProvider or string slug")

    @classmethod
    def _ensure_builtin(cls, key: str) -> None:
        if key in cls._factories:
            return
        spec = _BUILTIN_SPEECH_ADAPTERS.get(key)
        if spec is None:
            return
        module_path, class_name = spec
        module = importlib.import_module(module_path)
        adapter_cls = module.__dict__[class_name]

        def factory(**kwargs: Any) -> SpeechAdapter:
            return adapter_cls(**kwargs)

        cls._factories[key] = factory

    @classmethod
    def register(
        cls,
        provider: str | SpeechProvider,
        factory: Callable[..., SpeechAdapter],
        *,
        override: bool = False,
    ) -> None:
        key = cls._normalize_provider(provider)
        if key in cls._factories and not override:
            raise ValueError(f"Speech adapter already registered for provider='{key}'")
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider: str | SpeechProvider, **kwargs: Any) -> SpeechAdapter:
        key = cls._normalize_provider(provider)
        cls._ensure_builtin(key)
        if key not in cls._factories:
            raise ValueError(f"Speech adapter not registered for provider='{key}'")
        adapter = cls._factories[key](**kwargs)
        if not isinstance(adapter, SpeechAdapter):
            raise TypeError(
                f"Factory for provider='{key}' returned {type(adapter)!r}, expected SpeechAdapter"
            )
        adapter.validate()
        return adapter

    @classmethod
    def registered_providers(cls) -> list[str]:
        keys = set(cls._factories.keys()) | set(_BUILTIN_SPEECH_ADAPTERS.keys())
        return sorted(keys)
