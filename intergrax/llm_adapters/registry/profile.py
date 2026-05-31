# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Declarative Tier-3 LLM provider selection (mirrors IntegrationProfile pattern)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

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

    def create_adapter(self, **overrides: Any) -> LLMAdapter:
        from intergrax.llm_adapters.llm_provider_registry import LLMAdapterRegistry

        kwargs = {**self.options, **overrides}
        if self.model:
            kwargs.setdefault("model", self.model)
        return LLMAdapterRegistry.create(self.provider, **kwargs)

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
