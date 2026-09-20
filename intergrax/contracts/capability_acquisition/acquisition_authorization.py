# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Acquisition governance authorization port (UCA-3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.runtime_policy import PolicyDecision

_PERMITTED_POLICY_ACTIONS = frozenset({PolicyAction.ALLOW})
_BLOCKED_POLICY_ACTIONS = frozenset({PolicyAction.DENY})
_HITL_POLICY_ACTIONS = frozenset(
    {PolicyAction.REQUIRE_HUMAN, PolicyAction.ESCALATE},
)


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

    @model_validator(mode="after")
    def _outcome_aligns_with_policy_decision(
        self,
    ) -> CapabilityAcquisitionAuthorizationResult:
        if self.policy_decision is None:
            return self
        action = self.policy_decision.action
        if self.outcome is CapabilityAcquisitionAuthorizationOutcome.PERMITTED:
            if action not in _PERMITTED_POLICY_ACTIONS:
                raise ValueError(
                    "PERMITTED authorization requires policy action ALLOW",
                )
        elif self.outcome is CapabilityAcquisitionAuthorizationOutcome.BLOCKED:
            if action not in _BLOCKED_POLICY_ACTIONS:
                raise ValueError(
                    "BLOCKED authorization requires policy action DENY",
                )
        elif self.outcome is CapabilityAcquisitionAuthorizationOutcome.REQUIRES_HITL:
            if action not in _HITL_POLICY_ACTIONS:
                raise ValueError(
                    "REQUIRES_HITL authorization requires policy action "
                    "REQUIRE_HUMAN or ESCALATE",
                )
        return self


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
