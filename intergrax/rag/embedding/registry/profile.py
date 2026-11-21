# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Declarative embedding provider + model selection (mirrors LLMProfile pattern)."""

from __future__ import annotations

import os

from pydantic import BaseModel, ConfigDict, field_validator

from intergrax.integrations.contracts.base import LabSelectorOptIn
from intergrax.rag.embedding.contracts.runtime_binding import EmbeddingProviderConfigurationError
from intergrax.rag.embedding.registry.provider_authority import validate_embedding_provider_slug


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
        return validate_embedding_provider_slug(value.strip().lower())

    @field_validator("model", mode="before")
    @classmethod
    def _normalize_model(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("model must be a string or None")
        stripped = value.strip()
        return stripped or None


def embedding_profile_from_env(
    *,
    prefix: str = "INTERGRAX_EMBEDDING",
    lab_opt_in: LabSelectorOptIn | None = None,
) -> EmbeddingProfile:
    """
    Build profile from environment variables:

    - ``{PREFIX}_PROVIDER`` (required unless ``lab_opt_in`` is passed)
    - ``{PREFIX}_MODEL`` (optional)
    """
    provider_raw = os.getenv(f"{prefix}_PROVIDER")
    model_raw = os.getenv(f"{prefix}_MODEL")
    model = model_raw.strip() if model_raw else None
    if provider_raw is None or not provider_raw.strip():
        if lab_opt_in is None:
            raise EmbeddingProviderConfigurationError(
                f"{prefix}_PROVIDER is unset and no embedding provider was selected"
            )
        provider_raw = lab_opt_in.selector
    return EmbeddingProfile(provider=provider_raw, model=model or None)
