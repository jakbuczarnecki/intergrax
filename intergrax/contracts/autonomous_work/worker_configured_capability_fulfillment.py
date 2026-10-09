# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CONFIGURE_EXISTING fulfillment contracts (TRACE-X-P5-R2-P3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work.capability_acquisition import (
    WorkerCapabilityAcquisitionDecision,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
)
from intergrax.contracts.autonomous_work.worker_capability_recovery import (
    WorkerCapabilityRecoveryOutcome,
)
from intergrax.integrations.contracts.execution_integration_configuration import (
    ExecutionIntegrationConfigurationAdoption,
)


class WorkerConfiguredCapabilityFulfillmentFailureReason(StrEnum):
    INVALID_DECISION = "INVALID_DECISION"
    CONFIGURED_FULFILLMENT_UNAVAILABLE = "CONFIGURED_FULFILLMENT_UNAVAILABLE"
    OPPORTUNITY_NOT_FOUND = "OPPORTUNITY_NOT_FOUND"
    OPPORTUNITY_TENANT_MISMATCH = "OPPORTUNITY_TENANT_MISMATCH"
    PRINCIPAL_MISSING = "PRINCIPAL_MISSING"
    PRINCIPAL_TENANT_MISMATCH = "PRINCIPAL_TENANT_MISMATCH"
    REALIZATION_DENIED = "REALIZATION_DENIED"
    REALIZATION_FAILED = "REALIZATION_FAILED"
    BINDING_IDENTITY_MISMATCH = "BINDING_IDENTITY_MISMATCH"


@dataclass(frozen=True, slots=True)
class WorkerConfiguredCapabilityFulfillmentResult:
    """Explicit adoption handoff — AW stops before effective provider materialization."""

    adoption: ExecutionIntegrationConfigurationAdoption | None
    failure_reason: WorkerConfiguredCapabilityFulfillmentFailureReason | None = None
    correlation_ref: str | None = None

    def __post_init__(self) -> None:
        if self.adoption is not None and self.failure_reason is not None:
            raise ValueError("adoption and failure_reason are mutually exclusive")
        if self.adoption is None and self.failure_reason is None:
            raise ValueError("result requires adoption or failure_reason")


@runtime_checkable
class WorkerConfiguredCapabilityFulfillmentPort(Protocol):
    """Narrow INT-CONFIG fulfillment seam for CONFIGURE_EXISTING decisions."""

    def fulfill_configure_existing(
        self,
        request: WorkerCapabilityFulfillmentRequest,
        recovery: WorkerCapabilityRecoveryOutcome,
        decision: WorkerCapabilityAcquisitionDecision,
    ) -> WorkerConfiguredCapabilityFulfillmentResult: ...


__all__ = [
    "WorkerConfiguredCapabilityFulfillmentFailureReason",
    "WorkerConfiguredCapabilityFulfillmentPort",
    "WorkerConfiguredCapabilityFulfillmentResult",
]
