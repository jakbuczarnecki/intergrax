# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed default acquisition strategy selection (UCA-3)."""

from __future__ import annotations

from intergrax.contracts.capability_acquisition.acquisition_reason_code import (
    CapabilityAcquisitionReasonCode,
)
from intergrax.contracts.capability_acquisition.acquisition_request import (
    CapabilityAcquisitionRequest,
)
from intergrax.contracts.capability_acquisition.strategy_descriptor import (
    CapabilityAcquisitionStrategyDescriptor,
)
from intergrax.contracts.capability_acquisition.strategy_selection import (
    CapabilityAcquisitionGovernanceContext,
    CapabilityAcquisitionStrategySelection,
    CapabilityAcquisitionStrategySelectionOutcome,
)


class DefaultCapabilityAcquisitionStrategySelectionPolicy:
    """Deterministic safe selection — no hidden registration order preference."""

    def select(
        self,
        *,
        request: CapabilityAcquisitionRequest,
        candidates: tuple[CapabilityAcquisitionStrategyDescriptor, ...],
        governance_context: CapabilityAcquisitionGovernanceContext,
    ) -> CapabilityAcquisitionStrategySelection:
        del request, governance_context
        if not candidates:
            return CapabilityAcquisitionStrategySelection(
                outcome=CapabilityAcquisitionStrategySelectionOutcome.NO_STRATEGY,
                reason_code=CapabilityAcquisitionReasonCode.NO_STRATEGY,
                reason_detail="no eligible acquisition strategy",
            )
        if len(candidates) == 1:
            return CapabilityAcquisitionStrategySelection(
                outcome=CapabilityAcquisitionStrategySelectionOutcome.SELECTED,
                strategy_id=candidates[0].strategy_id,
                reason_code=CapabilityAcquisitionReasonCode.NONE,
            )
        strategy_ids = ", ".join(sorted(c.strategy_id for c in candidates))
        return CapabilityAcquisitionStrategySelection(
            outcome=CapabilityAcquisitionStrategySelectionOutcome.CONFLICT,
            reason_code=CapabilityAcquisitionReasonCode.SELECTION_CONFLICT,
            reason_detail=f"ambiguous strategies: {strategy_ids}",
        )


__all__ = ["DefaultCapabilityAcquisitionStrategySelectionPolicy"]
