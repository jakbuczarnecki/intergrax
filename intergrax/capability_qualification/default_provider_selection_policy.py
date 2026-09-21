# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Fail-closed default qualification provider selection (UCA-4)."""

from __future__ import annotations

from intergrax.contracts.capability_qualification.provider_descriptor import (
    CapabilityQualificationProviderDescriptor,
)
from intergrax.contracts.capability_qualification.provider_selection import (
    CapabilityQualificationProviderSelection,
    CapabilityQualificationProviderSelectionOutcome,
)
from intergrax.contracts.capability_qualification.qualification_reason_code import (
    CapabilityQualificationReasonCode,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)


class DefaultCapabilityQualificationProviderSelectionPolicy:
    """Deterministic safe selection — no hidden registration order preference."""

    def select(
        self,
        *,
        request: CapabilityQualificationRequest,
        candidates: tuple[CapabilityQualificationProviderDescriptor, ...],
    ) -> CapabilityQualificationProviderSelection:
        del request
        if not candidates:
            return CapabilityQualificationProviderSelection(
                outcome=CapabilityQualificationProviderSelectionOutcome.NO_PROVIDER,
                reason_code=CapabilityQualificationReasonCode.NO_PROVIDER,
                reason_detail="no eligible qualification provider",
            )
        if len(candidates) == 1:
            return CapabilityQualificationProviderSelection(
                outcome=CapabilityQualificationProviderSelectionOutcome.SELECTED,
                provider_id=candidates[0].provider_id,
                reason_code=CapabilityQualificationReasonCode.NONE,
            )
        provider_ids = ", ".join(sorted(c.provider_id for c in candidates))
        return CapabilityQualificationProviderSelection(
            outcome=CapabilityQualificationProviderSelectionOutcome.CONFLICT,
            reason_code=CapabilityQualificationReasonCode.SELECTION_CONFLICT,
            reason_detail=f"ambiguous providers: {provider_ids}",
        )


__all__ = ["DefaultCapabilityQualificationProviderSelectionPolicy"]
