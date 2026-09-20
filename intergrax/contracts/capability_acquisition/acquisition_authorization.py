# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Acquisition governance authorization port (UCA-3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.runtime_policy import PolicyDecision


class CapabilityAcquisitionAuthorizationOutcome(StrEnum):
    """Governance gate before any strategy side effect."""

    PERMITTED = "permitted"
    BLOCKED = "blocked"
    REQUIRES_HITL = "requires_hitl"


class CapabilityAcquisitionAuthorizationResult(BaseModel):
    """Typed authorization decision — strategies must not self-grant authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    outcome: CapabilityAcquisitionAuthorizationOutcome
    decision_id: str = ""
    policy_decision: PolicyDecision | None = None
    reason_detail: str = Field(default="")


@runtime_checkable
class CapabilityAcquisitionAuthorizationPort(Protocol):
    """Authorize acquisition before strategy.acquire() — reuse governance vocabulary."""

    def authorize(
        self,
        request: CapabilityAcquisitionRequest,
        eligible: tuple[CapabilityAcquisitionStrategyDescriptor, ...],
    ) -> CapabilityAcquisitionAuthorizationResult:
        """Evaluate whether acquisition may proceed for eligible strategies."""
        ...


__all__ = [
    "CapabilityAcquisitionAuthorizationOutcome",
    "CapabilityAcquisitionAuthorizationPort",
    "CapabilityAcquisitionAuthorizationResult",
]
