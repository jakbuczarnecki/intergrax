# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Strategy selection policy contracts (UCA-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.capability_catalog._validation import require_non_empty_text

SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_SELECTION_V1: Final = (
    "capability_acquisition_strategy_selection.v1"
)
_NON_EMPTY = Field(min_length=1)


class CapabilityAcquisitionStrategySelectionOutcome(StrEnum):
    """Typed selection decision — not an optional strategy reference."""

    SELECTED = "selected"
    NO_STRATEGY = "no_strategy"
    CONFLICT = "conflict"
    BLOCKED = "blocked"
    REQUIRES_HITL = "requires_hitl"


@dataclass(frozen=True, slots=True)
class CapabilityAcquisitionGovernanceContext:
    """Minimal governance metadata for selection policies."""

    authorization_decision_id: str | None = None


class CapabilityAcquisitionStrategySelection(BaseModel):
    """Deterministic strategy selection outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["capability_acquisition_strategy_selection.v1"] = (
        SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_SELECTION_V1
    )
    outcome: CapabilityAcquisitionStrategySelectionOutcome
    strategy_id: str | None = None
    reason_code: CapabilityAcquisitionReasonCode
    reason_detail: str = ""

    @field_validator("strategy_id")
    @classmethod
    def _validate_strategy_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_non_empty_text(value, label="strategy_id")

    @model_validator(mode="after")
    def _outcome_strategy_id_invariant(
        self,
    ) -> CapabilityAcquisitionStrategySelection:
        if self.outcome is CapabilityAcquisitionStrategySelectionOutcome.SELECTED:
            if self.strategy_id is None:
                raise ValueError("SELECTED outcome requires strategy_id")
            if self.reason_code is not CapabilityAcquisitionReasonCode.NONE:
                raise ValueError("SELECTED outcome requires reason_code NONE")
        elif self.strategy_id is not None:
            raise ValueError(
                "strategy_id is only allowed when outcome is SELECTED",
            )
        return self


@runtime_checkable
class CapabilityAcquisitionStrategySelectionPolicy(Protocol):
    """Pluginable selection — orchestrator must not embed ranking preferences."""

    def select(
        self,
        *,
        request: CapabilityAcquisitionRequest,
        candidates: tuple[CapabilityAcquisitionStrategyDescriptor, ...],
        governance_context: CapabilityAcquisitionGovernanceContext,
    ) -> CapabilityAcquisitionStrategySelection:
        """Choose at most one strategy from eligible candidates."""
        ...


__all__ = [
    "CapabilityAcquisitionGovernanceContext",
    "CapabilityAcquisitionStrategySelection",
    "CapabilityAcquisitionStrategySelectionOutcome",
    "CapabilityAcquisitionStrategySelectionPolicy",
    "SCHEMA_CAPABILITY_ACQUISITION_STRATEGY_SELECTION_V1",
]
