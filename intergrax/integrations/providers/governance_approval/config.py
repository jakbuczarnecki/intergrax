# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.runtime.integrations.categories._base import CategoryIntegrationConfig


class GovernanceApprovalIntegrationConfig(CategoryIntegrationConfig):
    """Typed non-secret configuration for Governance Approval HTTP access."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    base_url: str = Field(..., min_length=1, max_length=512)
    timeout_seconds: float = Field(default=5.0, gt=0, le=30)

    @field_validator("base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        cleaned = value.strip().rstrip("/")
        if not cleaned:
            raise ValueError("base_url_invalid")
        if not cleaned.startswith(("http://", "https://")):
            raise ValueError("base_url_scheme_invalid")
        return cleaned
