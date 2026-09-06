# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

from __future__ import annotations

from typing import Union

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider
from intergrax.llm_adapters.registry.registration_contract import (
    LLMAdapterDependencyError,
    LLMAdapterFactory,
    LLMAdapterRegistrationError,
    LLMAdapterRegistrationSpec,
    LLMAdapterRegistrationTarget,
    OptionalDependencyRequirement,
)

__all__ = [
    "LLMAdapterDependencyError",
    "LLMAdapterFactory",
    "LLMAdapterRegistrationError",
    "LLMAdapterRegistrationSpec",
    "LLMAdapterRegistrationTarget",
    "LLMAdapterRegistry",
    "OptionalDependencyRequirement",
]


class LLMAdapterRegistry:
    _factories: dict[str, LLMAdapterFactory] = {}
    _builtin_registrations_installed: bool = False

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
    def ensure_builtin_registrations_installed(cls) -> None:
        if cls._builtin_registrations_installed:
            return
        from intergrax.llm_adapters.providers.registrations.builtin import (
            register_builtin_llm_adapters,
        )

        register_builtin_llm_adapters(cls)
        cls._builtin_registrations_installed = True

    @classmethod
    def register_from_spec(
        cls,
        spec: LLMAdapterRegistrationSpec,
        *,
        override: bool = False,
    ) -> None:
        key = cls._normalize_provider(spec.provider_id)
        if key != spec.provider_id.strip().lower():
            raise LLMAdapterRegistrationError(
                f"LLM provider registration spec provider_id={spec.provider_id!r} "
                "must already be normalized."
            )
        if key in cls._factories and not override:
            raise ValueError(f"LLM adapter already registered for provider='{key}'")
        cls._factories[key] = spec.factory

    @classmethod
    def register(
        cls,
        provider: Union[str, LLMProvider, LLMAdapterRegistrationSpec],
        factory: LLMAdapterFactory | None = None,
        *,
        override: bool = False,
    ) -> None:
        if isinstance(provider, LLMAdapterRegistrationSpec):
            cls.register_from_spec(provider, override=override)
            return
        if factory is None:
            raise TypeError(
                "factory is required when provider is not LLMAdapterRegistrationSpec"
            )
        key = cls._normalize_provider(provider)
        cls.register_from_spec(
            LLMAdapterRegistrationSpec(provider_id=key, factory=factory),
            override=override,
        )

    @classmethod
    def create(cls, provider: Union[str, LLMProvider], **kwargs: object) -> LLMAdapter:
        key = cls._normalize_provider(provider)
        cls.ensure_builtin_registrations_installed()

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

        model_id = _resolve_adapter_model_id(adapter, kwargs)
        adapter = enrich_adapter_with_catalog_capabilities(
            adapter,
            provider=key,
            model=str(model_id) if model_id else None,
        )

        adapter.validate()
        return adapter

    @classmethod
    def registered_providers(cls) -> list[str]:
        cls.ensure_builtin_registrations_installed()
        return sorted(cls._factories.keys())


def _resolve_adapter_model_id(adapter: LLMAdapter, kwargs: dict[str, object]) -> str | None:
    model_kw = kwargs.get("model")
    if model_kw:
        return str(model_kw)
    try:
        model = adapter.model
    except AttributeError as exc:
        raise LLMAdapterRegistrationError(
            f"LLM adapter {type(adapter).__name__} violates LLMAdapter runtime "
            "contract: required public 'model' attribute is missing."
        ) from exc
    if not isinstance(model, str):
        raise TypeError(
            f"LLM adapter {type(adapter).__name__} violates LLMAdapter runtime "
            f"contract: 'model' must be str, got {type(model).__name__}."
        )
    return model if model else None
