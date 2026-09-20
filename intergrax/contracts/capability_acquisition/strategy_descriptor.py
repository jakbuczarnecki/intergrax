# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed strategy identity for governance and selection (UCA-3)."""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.capability_catalog._validation import require_non_empty_text
from intergrax.contracts.capability_catalog.kind import CapabilityKind

SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_DESCRIPTOR_V1: Final = (
    "capability_acquisition_strategy_descriptor.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityAcquisitionStrategyDescriptor(BaseModel):
    """Governance-facing strategy metadata — not a strategy implementation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_acquisition_strategy_descriptor.v1"] = (
        SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_DESCRIPTOR_V1
    )
    strategy_id: str = _NON_EMPTY
    supported_kinds: tuple[CapabilityKind, ...] = ()

    @field_validator("strategy_id")
    @classmethod
    def _validate_strategy_id(cls, value: str) -> str:
        return require_non_empty_text(value, label="strategy_id")


__all__ = [
    "SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_DESCRIPTOR_V1",
    "CapabilityAcquisitionStrategyDescriptor",
]
