# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Recovery orchestration → canonical worker capability fulfillment handoff (UCA-6C-R6-R5.8-H1)."""

from __future__ import annotations

from intergrax.autonomous_work.recovery_orchestration_ports import (
    PortAvailabilityDisposition,
    WorkerRecoveryCapabilityFulfillmentRequest,
    WorkerRecoveryCapabilityFulfillmentResult,
)
from intergrax.autonomous_work.worker_capability_fulfillment_ports import (
    WorkerCapabilityFulfillmentPort,
)


class WorkerRecoveryCapabilityFulfillmentService:
    """Caller layer — recovery orchestration depends on this port, not the coordinator."""

    def __init__(
        self,
        *,
        fulfillment: WorkerCapabilityFulfillmentPort,
    ) -> None:
        self._fulfillment = fulfillment

    def fulfill_recovery_capability(
        self,
        handoff: WorkerRecoveryCapabilityFulfillmentRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentResult:
        result = self._fulfillment.fulfill(handoff.fulfillment_request)
        return WorkerRecoveryCapabilityFulfillmentResult(
            disposition=PortAvailabilityDisposition.AVAILABLE,
            fulfillment_result=result,
        )


__all__ = ["WorkerRecoveryCapabilityFulfillmentService"]
