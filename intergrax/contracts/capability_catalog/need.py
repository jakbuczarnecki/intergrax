# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Neutral machine-facing capability need (ME-12) — not a discovery query."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind, V1_CAPABILITY_KINDS

SCHEMA_CAPABILITY_NEED_V1: Final = "capability_need.v1"


class CapabilityNeed(BaseModel):
    """Describes what capability is required — orthogonal to discovery query shape."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_need.v1"] = SCHEMA_CAPABILITY_NEED_V1
    need_id: str | None = None
    kinds: tuple[CapabilityKind, ...] = ()
    intent_summary: str | None = Field(default=None, min_length=1)

    @field_validator("need_id")
    @classmethod
    def _validate_need_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="need_id")

    @field_validator("kinds")
    @classmethod
    def _validate_kinds(cls, value: tuple[CapabilityKind, ...]) -> tuple[CapabilityKind, ...]:
        if not value:
            return value
        unknown = frozenset(value) - V1_CAPABILITY_KINDS
        if unknown:
            raise ValueError(
                f"unsupported capability kinds: {sorted(item.value for item in unknown)}",
            )
        return tuple(dict.fromkeys(value))


__all__ = ["CapabilityNeed", "SCHEMA_CAPABILITY_NEED_V1"]
