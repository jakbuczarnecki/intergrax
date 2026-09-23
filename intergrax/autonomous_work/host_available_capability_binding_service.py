# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Host-available capability binding coordination (UCA-6C-R6-R5.8-H1)."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.autonomous_work.worker_host_available_capability_binding import (
    HostAvailableCapabilityBindingProvider,
    HostAvailableCapabilityBindingRequest,
    HostAvailableCapabilityBindingResult,
)
from intergrax.contracts.capability_qualification.qualified_capability_binding import (
    QualifiedCapabilityBindingOutcome,
    QualifiedCapabilityBindingReasonCode,
)


class HostAvailableCapabilityBindingService:
    """Dispatch host-available binding to plugin providers — no AW routing."""

    def __init__(
        self,
        providers: tuple[HostAvailableCapabilityBindingProvider, ...],
    ) -> None:
        self._providers = providers

    def bind(
        self,
        request: HostAvailableCapabilityBindingRequest,
    ) -> HostAvailableCapabilityBindingResult:
        started_at = request.requested_at or datetime.now(tz=UTC)
        for provider in self._providers:
            if provider.supports(request):
                result = provider.bind(request)
                if result.binding_operation_id != request.binding_operation_id:
                    completed_at = datetime.now(tz=UTC)
                    return HostAvailableCapabilityBindingResult(
                        binding_operation_id=request.binding_operation_id,
                        outcome=QualifiedCapabilityBindingOutcome.CONFLICT,
                        reason_code=QualifiedCapabilityBindingReasonCode.INTEGRITY_CONFLICT,
                        started_at=started_at,
                        completed_at=completed_at,
                        reason_detail="binding_operation_id mismatch",
                    )
                return result
        completed_at = datetime.now(tz=UTC)
        return HostAvailableCapabilityBindingResult(
            binding_operation_id=request.binding_operation_id,
            outcome=QualifiedCapabilityBindingOutcome.NO_PROVIDER,
            reason_code=QualifiedCapabilityBindingReasonCode.PROVIDER_UNAVAILABLE,
            started_at=started_at,
            completed_at=completed_at,
        )


__all__ = ["HostAvailableCapabilityBindingService"]
