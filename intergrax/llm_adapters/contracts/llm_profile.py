# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Declarative Tier-3 LLM provider selection (mirrors IntegrationProfile pattern)."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.structured_json_value import validate_json_value_structure
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider, llm_provider_slug
from intergrax.llm_adapters.contracts.serialized_value import JsonValue

_RAW_CREDENTIAL_OPTIONS_ERROR = (
    "raw credentials are not allowed in LLMProfile.options; "
    "pass credentials via registry create_adapter(profile, secrets=...) or SecretsStore"
)

_FORBIDDEN_CREDENTIAL_OPTION_KEYS = frozenset({"api_key"})


def _validate_options_map(value: dict[str, object]) -> dict[str, JsonValue]:
    validated: dict[str, JsonValue] = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("LLMProfile.options keys must be non-empty strings")
        validated[key] = validate_json_value_structure(
            raw,
            field_name="LLMProfile.options",
        )
    return validated


class LLMProfile(BaseModel):
    """
    Typed LLM provider + model + constructor options for Tier-3 applications.

    Example::

        profile = LLMProfile(
            provider=LLMProvider.GROQ,
            model="llama-3.3-70b-versatile",
            options={"max_retries": 2},
        )
        llm = create_adapter(profile)
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False)

    provider: Union[LLMProvider, str]
    model: Optional[str] = None
    options: dict[str, JsonValue] = Field(default_factory=dict)
    fallback_profiles: tuple[LLMProfile, ...] = Field(default_factory=tuple)
    routing_policy_hint: str | None = None

    @field_validator("options", mode="before")
    @classmethod
    def _coerce_and_validate_options(cls, value: object) -> dict[str, JsonValue]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("LLMProfile.options must be a mapping")
        return _validate_options_map(value)

    @field_validator("options")
    @classmethod
    def _reject_raw_credentials_in_options(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
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
        return llm_provider_slug(provider)

    @classmethod
    def lab(cls) -> LLMProfile:
        """Laboratory default — local Ollama."""
        return cls(provider=LLMProvider.OLLAMA, model="llama3.1:latest")

    @classmethod
    def from_mapping(cls, data: Mapping[str, JsonValue]) -> LLMProfile:
        return cls.model_validate(dict(data))


__all__ = ["LLMProfile", "llm_provider_slug"]
