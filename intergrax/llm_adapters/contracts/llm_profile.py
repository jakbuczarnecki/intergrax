# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Declarative Tier-3 LLM provider selection (mirrors IntegrationProfile pattern)."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

_RAW_CREDENTIAL_OPTIONS_ERROR = (
    "raw credentials are not allowed in LLMProfile.options; "
    "pass credentials via create_adapter(secrets=...) or SecretsStore"
)

_FORBIDDEN_CREDENTIAL_OPTION_KEYS = frozenset({"api_key"})


class LLMProfile(BaseModel):
    """
    Typed LLM provider + model + constructor options for Tier-3 applications.

    Example::

        profile = LLMProfile(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            options={"max_retries": 2},
        )
        llm = profile.create_adapter()
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provider: Union[LLMProvider, str]
    model: Optional[str] = None
    options: dict[str, Any] = Field(default_factory=dict)
    fallback_profiles: tuple[LLMProfile, ...] = Field(default_factory=tuple)
    routing_policy_hint: str | None = None

    @field_validator("options")
    @classmethod
    def _reject_raw_credentials_in_options(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = _FORBIDDEN_CREDENTIAL_OPTION_KEYS.intersection(value)
        if forbidden:
            raise ValueError(_RAW_CREDENTIAL_OPTIONS_ERROR)
        return value

    @field_validator("fallback_profiles", mode="before")
    @classmethod
    def _coerce_fallback_profiles(cls, value: object) -> tuple[LLMProfile, ...]:
        if value is None:
            return ()
        if isinstance(value, LLMProfile):
            return (value,)
        if isinstance(value, list):
            return tuple(LLMProfile.model_validate(item) if isinstance(item, dict) else item for item in value)
        if isinstance(value, tuple):
            return value
        raise ValueError("fallback_profiles must be a sequence of LLMProfile")

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, value: str | LLMProvider) -> LLMProvider | str:
        if isinstance(value, LLMProvider):
            return value
        if isinstance(value, str) and value.strip():
            key = value.strip().lower()
            try:
                return LLMProvider(key)
            except ValueError:
                return key
        raise ValueError("provider must be a non-empty LLMProvider or registered string slug")

    @classmethod
    def _provider_slug(cls, provider: LLMProvider | str) -> str:
        if isinstance(provider, LLMProvider):
            return provider.value
        return str(provider).strip().lower()

    @classmethod
    def lab(cls) -> LLMProfile:
        """Laboratory default — local Ollama."""
        return cls(provider=LLMProvider.OLLAMA, model="llama3.1:latest")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> LLMProfile:
        return cls.model_validate(dict(data))

