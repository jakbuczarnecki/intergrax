# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Canonical installation slot scope: (environment_id, installation_slot_id)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class InstallationSlotScope(BaseModel):
    """Environment-owned active installation pointer scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    environment_id: str = _NON_EMPTY
    installation_slot_id: str = _NON_EMPTY

    @field_validator("environment_id", "installation_slot_id")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)
