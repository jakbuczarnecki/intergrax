# © Artur Czarnecki. All rights reserved.

"""Factory registry for ``SpeechAdapter`` implementations."""

from __future__ import annotations

import importlib
from typing import Any, Callable

from intergrax.speech_adapters.contracts.speech_adapter import SpeechAdapter

_BUILTIN_SPEECH_ADAPTERS: dict[str, tuple[str, str]] = {
    "stub": (
        "intergrax.speech_adapters.providers.stub_speech",
        "StubSpeechAdapter",
    ),
    "elevenlabs": (
        "intergrax.speech_adapters.providers.elevenlabs_speech",
        "ElevenLabsSpeechAdapter",
    ),
}


class SpeechAdapterRegistry:
    """Create in-process speech adapters by provider slug."""

    _factories: dict[str, Callable[..., SpeechAdapter]] = {}

    @staticmethod
    def _normalize_provider(provider_slug: str) -> str:
        if isinstance(provider_slug, str) and provider_slug.strip():
            return provider_slug.strip().lower()
        raise ValueError("provider_slug must be a non-empty string")

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
        provider_slug: str,
        factory: Callable[..., SpeechAdapter],
        *,
        override: bool = False,
    ) -> None:
        key = cls._normalize_provider(provider_slug)
        if key in cls._factories and not override:
            raise ValueError(f"Speech adapter already registered for provider_slug='{key}'")
        cls._factories[key] = factory

    @classmethod
    def create(cls, provider_slug: str, **kwargs: Any) -> SpeechAdapter:
        key = cls._normalize_provider(provider_slug)
        cls._ensure_builtin(key)
        if key not in cls._factories:
            raise ValueError(f"Speech adapter not registered for provider_slug='{key}'")
        adapter = cls._factories[key](**kwargs)
        if not isinstance(adapter, SpeechAdapter):
            raise TypeError(
                f"Factory for provider_slug='{key}' returned {type(adapter)!r}, expected SpeechAdapter"
            )
        adapter.validate()
        return adapter

    @classmethod
    def registered_providers(cls) -> list[str]:
        keys = set(cls._factories.keys()) | set(_BUILTIN_SPEECH_ADAPTERS.keys())
        return sorted(keys)

    @classmethod
    def is_registered(cls, provider_slug: str) -> bool:
        key = cls._normalize_provider(provider_slug)
        cls._ensure_builtin(key)
        return key in cls._factories
