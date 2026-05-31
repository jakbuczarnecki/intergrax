# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Declarative Tier-3 LLM provider selection (mirrors IntegrationProfile pattern)."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from intergrax.integrations.contracts.secrets_store import SecretsStore

from intergrax.llm_adapters.registry.secrets import (
    load_api_key_from_secrets_store,
    merge_secrets_into_options,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.llm_adapter import LLMAdapter
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider


class LLMProfile(BaseModel):
    """
    Typed LLM provider + model + constructor options for Tier-3 applications.

    Example::

        profile = LLMProfile(provider=LLMProvider.GROQ, model="llama-3.3-70b-versatile", max_retries=2)
        llm = profile.create_adapter()
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provider: LLMProvider
    model: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, value: str | LLMProvider) -> LLMProvider:
        if isinstance(value, LLMProvider):
            return value
        if isinstance(value, str) and value.strip():
            return LLMProvider(value.strip().lower())
        raise ValueError("provider must be a non-empty LLMProvider or string slug")

    def create_adapter(
        self,
        *,
        secrets: Optional[Mapping[str, str]] = None,
        **overrides: Any,
    ) -> LLMAdapter:
        from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

        kwargs = merge_secrets_into_options(self.provider, {**self.options, **overrides}, secrets)
        if self.model:
            kwargs.setdefault("model", self.model)
        return LLMAdapterRegistry.create(self.provider, **kwargs)

    def with_secrets(self, secrets: Mapping[str, str]) -> LLMProfile:
        """Return profile with secrets merged into ``options`` (api_key, etc.)."""
        return self.model_copy(
            update={"options": merge_secrets_into_options(self.provider, dict(self.options), secrets)}
        )

    def create_adapter_from_secrets_store(
        self,
        store: "SecretsStore",
        *,
        secret_path: str | None = None,
        **overrides: Any,
    ) -> LLMAdapter:
        """Resolve API key from Vault/secrets integration, then create adapter."""
        key = load_api_key_from_secrets_store(store, self.provider, path=secret_path)
        return self.with_secrets({"api_key": key}).create_adapter(**overrides)

    @classmethod
    def lab(cls) -> LLMProfile:
        """Laboratory default — local Ollama."""
        return cls(provider=LLMProvider.OLLAMA, model="llama3.1:latest")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LLMProfile:
        return cls.model_validate(dict(data))


def llm_profile_from_env(*, prefix: str = "INTERGRAX_LLM") -> LLMProfile:
    """
    Build profile from environment variables:

    - ``{PREFIX}_PROVIDER`` (required, e.g. ``groq``)
    - ``{PREFIX}_MODEL`` (optional)
    """
    provider_raw = os.getenv(f"{prefix}_PROVIDER", LLMProvider.OLLAMA.value).strip()
    model = os.getenv(f"{prefix}_MODEL")
    return LLMProfile(provider=LLMProvider(provider_raw), model=model or None)
