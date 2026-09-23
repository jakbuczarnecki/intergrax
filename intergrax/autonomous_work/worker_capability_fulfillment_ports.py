# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ports for worker capability fulfillment — no domain ownership in consumer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)


@runtime_checkable
class CapabilityRealizationCoordinatorPort(Protocol):
    """Canonical UCA-2 realization dispatch — opaque provider selection."""

    def realize(
        self,
        request: CapabilityRealizationRequest,
    ) -> CapabilityRealizationResult: ...


@runtime_checkable
class WorkerCapabilityDirectReuseFulfillmentPort(Protocol):
    """Host-available DIRECT_REUSE binding and execution — not consumer-local discovery."""

    def fulfill_direct_reuse(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        recovery: WorkerCapabilityRecoveryOutcome,
    ) -> WorkerCapabilityFulfillmentResult: ...


__all__ = [
    "CapabilityRealizationCoordinatorPort",
    "WorkerCapabilityDirectReuseFulfillmentPort",
]
