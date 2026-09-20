# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit permit-all acquisition authorization adapter (composition root)."""

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
    """Allow acquisition — must be injected explicitly; not a UCA core default."""

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
