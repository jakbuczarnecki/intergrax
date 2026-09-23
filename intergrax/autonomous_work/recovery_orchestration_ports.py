# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Extension ports for AW-6B recovery orchestration — fail-closed by default."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, TypeVar

from intergrax.contracts.autonomous_work.execution_dispatch import (
    WorkerExecutionDispatchRequest,
    WorkerExecutionDispatchResult,
)
from intergrax.contracts.autonomous_work.worker_capability_fulfillment import (
    WorkerCapabilityFulfillmentRequest,
    WorkerCapabilityFulfillmentResult,
)
from intergrax.contracts.autonomous_work.recovery_orchestration import (
    WorkerRecoveryEpisode,
    WorkerRecoveryOrchestrationRequest,
    WorkerRecoveryResumeTarget,
)
from intergrax.contracts.autonomous_work.references import HumanPendingReference
from intergrax.contracts.autonomous_work.worker import WorkerInstance
from intergrax.contracts.execution_identity import ExecutionId

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class PortAvailabilityDisposition(StrEnum):
    """Typed port availability outcome."""

    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class CanonicalExecutionTerminalDisposition(StrEnum):
    """Canonical execution terminal read outcome."""

    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class CanonicalExecutionTerminalOutcome:
    """Canonical execution terminal evidence consumed by recovery."""

    disposition: CanonicalExecutionTerminalDisposition
    execution_id: ExecutionId
    failure_ref: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerRecoveryReplanRequest:
    """Trusted alternate path preparation request."""

    episode: WorkerRecoveryEpisode
    alternative_path_ref: str


@dataclass(frozen=True, slots=True)
class WorkerRecoveryReplanResult:
    """Trusted alternate execution intent — no authority."""

    disposition: PortAvailabilityDisposition
    resume_target: WorkerRecoveryResumeTarget | None = None


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAcquisitionRequest:
    """Future-facing capability acquisition handoff."""

    episode: WorkerRecoveryEpisode
    capability_missing_ref: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerCapabilityAcquisitionResult:
    """Capability acquisition port outcome."""

    disposition: PortAvailabilityDisposition


@dataclass(frozen=True, slots=True)
class HumanDecisionRequest:
    """Canonical human decision correlation request."""

    episode: WorkerRecoveryEpisode
    human_decision_ref: HumanPendingReference
    requested_at: datetime


@dataclass(frozen=True, slots=True)
class HumanDecisionRequestResult:
    """Human decision port outcome."""

    disposition: PortAvailabilityDisposition
    correlation_ref: str | None = None


@dataclass(frozen=True, slots=True)
class WorkerEscalationRequest:
    """Typed escalation intent — no direct channel integration."""

    episode: WorkerRecoveryEpisode
    reason: str


@dataclass(frozen=True, slots=True)
class WorkerEscalationResult:
    """Escalation port outcome."""

    disposition: PortAvailabilityDisposition
    escalation_ref: str | None = None


class WorkerRecoveryReplanPort(Protocol):
    def prepare_alternative(
        self,
        request: WorkerRecoveryReplanRequest,
    ) -> WorkerRecoveryReplanResult: ...


class WorkerCapabilityAcquisitionPort(Protocol):
    def request_acquisition(
        self,
        request: WorkerCapabilityAcquisitionRequest,
    ) -> WorkerCapabilityAcquisitionResult: ...


@dataclass(frozen=True, slots=True)
class WorkerRecoveryCapabilityFulfillmentRequest:
    """Episode-scoped fulfillment handoff — builder supplies canonical fulfillment request."""

    episode: WorkerRecoveryEpisode
    orchestration_request: WorkerRecoveryOrchestrationRequest
    fulfillment_request: WorkerCapabilityFulfillmentRequest


@dataclass(frozen=True, slots=True)
class WorkerRecoveryCapabilityFulfillmentResult:
    """Fulfillment port outcome for recovery orchestration."""

    disposition: PortAvailabilityDisposition
    fulfillment_result: WorkerCapabilityFulfillmentResult | None = None


class WorkerRecoveryCapabilityFulfillmentRequestBuilderPort(Protocol):
    """Maps durable recovery episode context to consumer fulfillment request."""

    def build_fulfillment_request(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        request: WorkerRecoveryOrchestrationRequest,
    ) -> WorkerCapabilityFulfillmentRequest | None: ...


class WorkerRecoveryCapabilityFulfillmentPort(Protocol):
    """Canonical worker capability fulfillment entry for recovery orchestration."""

    def fulfill_recovery_capability(
        self,
        handoff: WorkerRecoveryCapabilityFulfillmentRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentResult: ...


class CanonicalExecutionOutcomeReader(Protocol):
    def get_terminal_outcome(
        self,
        execution_id: ExecutionId,
    ) -> CanonicalExecutionTerminalOutcome: ...


class HumanDecisionRequestPort(Protocol):
    def request_human_decision(
        self,
        request: HumanDecisionRequest,
    ) -> HumanDecisionRequestResult: ...


class WorkerEscalationPort(Protocol):
    def escalate(
        self,
        request: WorkerEscalationRequest,
    ) -> WorkerEscalationResult: ...


class WorkerRecoveryExecutionDispatchPort(Protocol[InputT, OutputT]):
    """Dispatch recovery execution through AW-5A — no direct runtime access."""

    async def dispatch_recovery(
        self,
        *,
        episode: WorkerRecoveryEpisode,
        worker: WorkerInstance,
        resume_target: WorkerRecoveryResumeTarget,
        attempt_number: int,
        request: WorkerExecutionDispatchRequest[InputT, OutputT],
    ) -> WorkerExecutionDispatchResult[OutputT]: ...


class UnavailableWorkerRecoveryReplanPort:
    def prepare_alternative(
        self,
        request: WorkerRecoveryReplanRequest,
    ) -> WorkerRecoveryReplanResult:
        return WorkerRecoveryReplanResult(
            disposition=PortAvailabilityDisposition.UNAVAILABLE
        )


class UnavailableWorkerCapabilityAcquisitionPort:
    def request_acquisition(
        self,
        request: WorkerCapabilityAcquisitionRequest,
    ) -> WorkerCapabilityAcquisitionResult:
        return WorkerCapabilityAcquisitionResult(
            disposition=PortAvailabilityDisposition.UNAVAILABLE,
        )


class UnavailableWorkerRecoveryCapabilityFulfillmentPort:
    def fulfill_recovery_capability(
        self,
        handoff: WorkerRecoveryCapabilityFulfillmentRequest,
    ) -> WorkerRecoveryCapabilityFulfillmentResult:
        return WorkerRecoveryCapabilityFulfillmentResult(
            disposition=PortAvailabilityDisposition.UNAVAILABLE,
        )


class UnavailableHumanDecisionRequestPort:
    def request_human_decision(
        self,
        request: HumanDecisionRequest,
    ) -> HumanDecisionRequestResult:
        return HumanDecisionRequestResult(
            disposition=PortAvailabilityDisposition.UNAVAILABLE
        )


class UnavailableWorkerEscalationPort:
    def escalate(
        self,
        request: WorkerEscalationRequest,
    ) -> WorkerEscalationResult:
        return WorkerEscalationResult(
            disposition=PortAvailabilityDisposition.UNAVAILABLE
        )


class UnavailableCanonicalExecutionOutcomeReader:
    def get_terminal_outcome(
        self,
        execution_id: ExecutionId,
    ) -> CanonicalExecutionTerminalOutcome:
        return CanonicalExecutionTerminalOutcome(
            disposition=CanonicalExecutionTerminalDisposition.UNAVAILABLE,
            execution_id=execution_id,
        )
