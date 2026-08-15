# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical application-environment administrative scope (AP-11-FIX-2)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ApplicationEnvironmentIdentity(BaseModel):
    """Immutable composite scope: (application_id, application_environment_id)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY

    @field_validator("application_id", "application_environment_id")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)
