# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Composition-default acquisition authorization (UCA-3)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_authorization import (
    CapabilityAcquisitionAuthorizationOutcome,
    CapabilityAcquisitionAuthorizationResult,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.policy_action import PolicyAction
from intergrax.contracts.runtime_policy import PolicyDecision


class PermitCapabilityAcquisitionAuthorizationPort:
    """Allow acquisition when no external governance adapter is wired."""

    def authorize(
        self,
        request: CapabilityAcquisitionRequest,
        eligible: tuple[CapabilityAcquisitionStrategyDescriptor, ...],
    ) -> CapabilityAcquisitionAuthorizationResult:
        del request, eligible
        return CapabilityAcquisitionAuthorizationResult(
            outcome=CapabilityAcquisitionAuthorizationOutcome.PERMITTED,
            decision_id="uca3.default.permit",
            policy_decision=PolicyDecision(action=PolicyAction.ALLOW),
        )


__all__ = ["PermitCapabilityAcquisitionAuthorizationPort"]
