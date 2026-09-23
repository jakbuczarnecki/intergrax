# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Ports for worker capability fulfillment — no domain ownership in consumer."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityAcquisitionRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
)
from intergrax.contracts.autonomous_work.worker_qualified_capability_resume import (
    WorkerQualifiedCapabilityResumeRequest,
    WorkerQualifiedCapabilityResumeResult,
)
from intergrax.contracts.capability_acquisition.request import (
    CapabilityRealizationRequest,
)
from intergrax.contracts.capability_acquisition.result import (
    CapabilityRealizationResult,
)


@runtime_checkable
class WorkerCapabilityRecoveryPort(Protocol):
    """Canonical recovery sequencing — discovery, acquisition, qualification."""

    def coordinate_recovery(
        self,
        request: WorkerCapabilityAcquisitionRequest,
        *,
        decided_at: datetime | None = None,
        allow_generic_acquisition: bool = True,
    ) -> WorkerCapabilityRecoveryOutcome: ...


@runtime_checkable
class WorkerQualifiedCapabilityResumePort(Protocol):
    """Post-qualification binding and execution handoff."""

    def resume(
        self,
        request: WorkerQualifiedCapabilityResumeRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerQualifiedCapabilityResumeResult: ...


@runtime_checkable
class WorkerCapabilityFulfillmentPort(Protocol):
    """Consumer fulfillment seam — orchestration only."""

    def fulfill(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        *,
        decided_at: datetime | None = None,
    ) -> WorkerCapabilityFulfillmentResult: ...


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
    "WorkerCapabilityFulfillmentPort",
    "WorkerCapabilityRecoveryPort",
    "WorkerQualifiedCapabilityResumePort",
]
