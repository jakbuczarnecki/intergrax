# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Declarative embedding provider + model selection (mirrors LLMProfile pattern)."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, field_validator

_REGISTERED_EMBEDDING_PROVIDERS = frozenset(
    {"hf", "openai", "ollama", "vllm", "llama_cpp"}
)
_DEFAULT_EMBEDDING_PROVIDER = "ollama"


class EmbeddingProfile(BaseModel):
    """Typed embedding provider + model for runtime bootstrap."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str | None = None

    @field_validator("provider", mode="before")
    @classmethod
    def _coerce_provider(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("provider must be a non-empty string slug")
        key = value.strip().lower()
        if key not in _REGISTERED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"unknown embedding provider slug {key!r}; "
                f"expected one of {sorted(_REGISTERED_EMBEDDING_PROVIDERS)}"
            )
        return key

    @field_validator("model", mode="before")
    @classmethod
    def _normalize_model(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("model must be a string or None")
        stripped = value.strip()
        return stripped or None


def embedding_profile_from_env(*, prefix: str = "INTERGRAX_EMBEDDING") -> EmbeddingProfile:
    """
    Build profile from environment variables:

    - ``{PREFIX}_PROVIDER`` (default ``ollama``)
    - ``{PREFIX}_MODEL`` (optional)
    """
    provider_raw = os.getenv(f"{prefix}_PROVIDER", _DEFAULT_EMBEDDING_PROVIDER)
    model_raw = os.getenv(f"{prefix}_MODEL")
    model = model_raw.strip() if model_raw else None
    return EmbeddingProfile(provider=provider_raw, model=model or None)
