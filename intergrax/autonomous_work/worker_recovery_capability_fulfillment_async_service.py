# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Async recovery orchestration → canonical worker capability fulfillment (UCA-6C-R6-R5.8-R2-H1)."""

from __future__ import annotations

from intergrax.autonomous_work.recovery_orchestration_ports import (
    PortAvailabilityDisposition,
    WorkerRecoveryCapabilityFulfillmentRequest,
    WorkerRecoveryCapabilityFulfillmentResult,
)
from intergrax.autonomous_work.worker_capability_fulfillment_coordinator import (
    WorkerCapabilityFulfillmentCoordinator,
)


class WorkerRecoveryCapabilityFulfillmentAsyncService:
    """Async caller layer — uses async qualified execution without sync run_async."""

    def __init__(
        self,
        *,
        fulfillment: WorkerCapabilityFulfillmentCoordinator,
    ) -> None:
        self._fulfillment = fulfillment

    async def fulfill_recovery_capability_async(
        self,
        handoff: WorkerRecoveryCapabilityFulfillmentRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentResult:
        result = await self._fulfillment.fulfill_async(handoff.fulfillment_request)
        return WorkerRecoveryCapabilityFulfillmentResult(
            disposition=PortAvailabilityDisposition.AVAILABLE,
            fulfillment_result=result,
        )


__all__ = ["WorkerRecoveryCapabilityFulfillmentAsyncService"]
