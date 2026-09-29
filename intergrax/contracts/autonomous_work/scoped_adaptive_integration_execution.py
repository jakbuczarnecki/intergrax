# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""AW-7C-P4 scoped adaptive integration execution contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from intergrax.contracts.autonomous_work._validation import (
    freeze_tuple,
    require_aware_utc,
    require_non_empty_text,
)
from intergrax.contracts.autonomous_work.execution_dispatch import WorkerExecutionSource
from intergrax.contracts.autonomous_work.ids import (
    WorkerInstanceId,
    validate_worker_instance_id,
)
from intergrax.contracts.autonomous_work.revision import Revision, validate_revision
from intergrax.contracts.autonomous_work.scoped_adaptive_integration import (
    ScopedAdaptiveIntegrationPreparationResult,
    ScopedAdaptiveIntegrationPreparationStatus,
)
from intergrax.contracts.capability_qualification.lifecycle_decision import (
    CapabilityQualificationLifecycleOutcome,
)
from intergrax.contracts.capability_qualification.qualification_decision import (
    CapabilityQualificationDecision,
)
from intergrax.contracts.capability_qualification.qualification_outcome import (
    CapabilityQualificationOutcome,
)
from intergrax.contracts.capability_qualification.qualification_request import (
    CapabilityQualificationRequest,
)
from intergrax.contracts.capability_qualification.qualification_subject import (
    CapabilityQualificationSubject,
)
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.contracts.sandbox_network_egress import NetworkEgressAllowlist
from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.scoped_integration_adaptation import (
    ScopedIntegrationAdaptationArtifact,
    ScopedIntegrationAdaptationOperationId,
)


class ScopedAdaptiveIntegrationExecutionOutcome(StrEnum):
    QUALIFICATION_REJECTED = "QUALIFICATION_REJECTED"
    GOVERNANCE_DENIED = "GOVERNANCE_DENIED"
    GOVERNANCE_UNAVAILABLE = "GOVERNANCE_UNAVAILABLE"
    CREDENTIAL_DENIED = "CREDENTIAL_DENIED"
    SANDBOX_SECURITY_UNSATISFIED = "SANDBOX_SECURITY_UNSATISFIED"
    EXECUTION_FAILED = "EXECUTION_FAILED"
    DUPLICATE_INVOCATION = "DUPLICATE_INVOCATION"
    EXECUTED = "EXECUTED"


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionHandoff:
    """Safe execution intent — not permission or secret material."""

    artifact: ScopedIntegrationAdaptationArtifact
    qualification_subject: CapabilityQualificationSubject
    accepted_qualification: CapabilityQualificationDecision
    qualification_request_id: str
    requested_operation: ScopedIntegrationAdaptationOperationId
    tenant_id: str
    integration_category: IntegrationCategory
    provider_id: str
    resource_scope: str
    scope_fingerprint: str
    permitted_operations: tuple[ScopedIntegrationAdaptationOperationId, ...]
    network_allowlist: NetworkEgressAllowlist
    credential_grant_ref: str
    correlation_id: str | None
    causation_id: str | None
    execution_idempotency_key: str

    def __post_init__(self) -> None:
        if type(self.artifact) is not ScopedIntegrationAdaptationArtifact:
            raise TypeError("artifact must be ScopedIntegrationAdaptationArtifact")
        if type(self.qualification_subject) is not CapabilityQualificationSubject:
            raise TypeError("qualification_subject must be CapabilityQualificationSubject")
        if not isinstance(self.accepted_qualification, CapabilityQualificationDecision):
            raise TypeError("accepted_qualification must be CapabilityQualificationDecision")
        if type(self.requested_operation) is not ScopedIntegrationAdaptationOperationId:
            raise TypeError("requested_operation must be ScopedIntegrationAdaptationOperationId")
        object.__setattr__(
            self,
            "qualification_request_id",
            require_non_empty_text(
                self.qualification_request_id,
                label="qualification_request_id",
            ),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "provider_id",
            require_non_empty_text(self.provider_id, label="provider_id"),
        )
        object.__setattr__(
            self,
            "resource_scope",
            require_non_empty_text(self.resource_scope, label="resource_scope"),
        )
        object.__setattr__(
            self,
            "scope_fingerprint",
            require_non_empty_text(self.scope_fingerprint, label="scope_fingerprint"),
        )
        object.__setattr__(
            self,
            "permitted_operations",
            freeze_tuple(self.permitted_operations, label="permitted_operations"),
        )
        if self.requested_operation not in self.permitted_operations:
            raise ValueError("requested_operation must be in permitted_operations")
        if type(self.network_allowlist) is not NetworkEgressAllowlist:
            raise TypeError("network_allowlist must be NetworkEgressAllowlist")
        object.__setattr__(
            self,
            "credential_grant_ref",
            require_non_empty_text(
                self.credential_grant_ref,
                label="credential_grant_ref",
            ),
        )
        object.__setattr__(
            self,
            "execution_idempotency_key",
            require_non_empty_text(
                self.execution_idempotency_key,
                label="execution_idempotency_key",
            ),
        )


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationWorkerDispatchContext:
    worker_revision: Revision
    requested_scopes: tuple[str, ...]
    source: WorkerExecutionSource

    def __post_init__(self) -> None:
        if type(self.worker_revision) is not Revision:
            raise TypeError("worker_revision must be Revision")
        validate_revision(self.worker_revision)
        if type(self.source) is not WorkerExecutionSource:
            raise TypeError("source must be WorkerExecutionSource")
        object.__setattr__(
            self,
            "requested_scopes",
            freeze_tuple(self.requested_scopes, label="requested_scopes"),
        )


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionRequest:
    preparation: ScopedAdaptiveIntegrationPreparationResult
    worker_dispatch: ScopedAdaptiveIntegrationWorkerDispatchContext
    tenant_id: str
    requested_operation: ScopedIntegrationAdaptationOperationId
    execution_idempotency_key: str
    requested_at: datetime

    def __post_init__(self) -> None:
        if type(self.preparation) is not ScopedAdaptiveIntegrationPreparationResult:
            raise TypeError("preparation must be ScopedAdaptiveIntegrationPreparationResult")
        if type(self.worker_dispatch) is not ScopedAdaptiveIntegrationWorkerDispatchContext:
            raise TypeError("worker_dispatch must be ScopedAdaptiveIntegrationWorkerDispatchContext")
        if type(self.requested_operation) is not ScopedIntegrationAdaptationOperationId:
            raise TypeError("requested_operation must be ScopedIntegrationAdaptationOperationId")
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        object.__setattr__(
            self,
            "execution_idempotency_key",
            require_non_empty_text(
                self.execution_idempotency_key,
                label="execution_idempotency_key",
            ),
        )
        object.__setattr__(
            self,
            "requested_at",
            require_aware_utc(self.requested_at, label="requested_at"),
        )


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionRuntimeEnvelope:
    """Execution-runtime result — domain outcome plus optional operation output."""

    outcome: ScopedAdaptiveIntegrationExecutionOutcome
    output: ScopedAdaptiveIntegrationExecutionOutput | None = None
    error_detail: str = ""

    def __post_init__(self) -> None:
        if type(self.outcome) is not ScopedAdaptiveIntegrationExecutionOutcome:
            raise TypeError("outcome must be ScopedAdaptiveIntegrationExecutionOutcome")
        if self.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED:
            if self.output is None:
                raise ValueError("EXECUTED requires output")
        elif self.output is not None:
            raise ValueError("non-EXECUTED must not carry output")


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionOutput:
    evidence_ref: str
    tenant_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evidence_ref",
            require_non_empty_text(self.evidence_ref, label="evidence_ref"),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )


@dataclass(frozen=True, slots=True)
class ScopedAdaptiveIntegrationExecutionResult:
    outcome: ScopedAdaptiveIntegrationExecutionOutcome
    worker_instance_id: WorkerInstanceId
    execution_idempotency_key: str
    tenant_id: str
    run_id: RunId | None = None
    attempt_id: AttemptId | None = None
    execution_id: ExecutionId | None = None
    handoff: ScopedAdaptiveIntegrationExecutionHandoff | None = None
    operation_output: ScopedAdaptiveIntegrationExecutionOutput | None = None
    error_detail: str = ""

    def __post_init__(self) -> None:
        validate_worker_instance_id(self.worker_instance_id)
        if type(self.outcome) is not ScopedAdaptiveIntegrationExecutionOutcome:
            raise TypeError("outcome must be ScopedAdaptiveIntegrationExecutionOutcome")
        object.__setattr__(
            self,
            "execution_idempotency_key",
            require_non_empty_text(
                self.execution_idempotency_key,
                label="execution_idempotency_key",
            ),
        )
        object.__setattr__(
            self,
            "tenant_id",
            require_non_empty_text(self.tenant_id, label="tenant_id"),
        )
        if self.run_id is not None:
            validate_run_id(self.run_id)
        if self.attempt_id is not None:
            validate_attempt_id(self.attempt_id)
        if self.execution_id is not None:
            validate_execution_id(self.execution_id)
        if self.outcome is ScopedAdaptiveIntegrationExecutionOutcome.EXECUTED:
            if self.run_id is None or self.attempt_id is None or self.execution_id is None:
                raise ValueError("EXECUTED requires canonical execution identity")
            if self.operation_output is None:
                raise ValueError("EXECUTED requires operation_output")


@runtime_checkable
class WorkerScopedAdaptiveIntegrationExecutionCoordinatorPort(Protocol):
    async def execute(
        self,
        request: ScopedAdaptiveIntegrationExecutionRequest,
    ) -> ScopedAdaptiveIntegrationExecutionResult: ...


def validate_preparation_artifact_subject_continuity(
    *,
    preparation: ScopedAdaptiveIntegrationPreparationResult,
    qualification_request: CapabilityQualificationRequest,
) -> str | None:
    if preparation.status is not ScopedAdaptiveIntegrationPreparationStatus.QUALIFICATION_PENDING:
        return "preparation not QUALIFICATION_PENDING"
    artifact = preparation.artifact
    subject = preparation.qualification_subject
    if artifact is None or subject is None:
        return "missing artifact or subject"
    if artifact.artifact_id != subject.subject_id:
        return "artifact id subject mismatch"
    if artifact.artifact_fingerprint != subject.subject_integrity_fingerprint:
        return "artifact fingerprint mismatch"
    if artifact.scope_fingerprint != subject.scope_fingerprint:
        return "scope fingerprint mismatch"
    if artifact.tenant_id != subject.tenant_id:
        return "artifact subject tenant mismatch"
    if qualification_request.subject != subject:
        return "qualification request subject mismatch"
    lineage = subject.adaptation_lineage
    if lineage is None:
        return "missing adaptation lineage"
    if lineage.artifact_id != artifact.artifact_id:
        return "lineage artifact mismatch"
    if lineage.strategy_id != artifact.strategy_id:
        return "lineage strategy mismatch"
    if lineage.provider_id != artifact.provider_id:
        return "lineage provider mismatch"
    if lineage.candidate_revision != artifact.candidate_revision:
        return "lineage revision mismatch"
    return None


def validate_qualification_decision_continuity(
    *,
    qualification_request: CapabilityQualificationRequest,
    decision: CapabilityQualificationDecision,
    artifact: ScopedIntegrationAdaptationArtifact,
    subject: CapabilityQualificationSubject,
) -> str | None:
    result = decision.qualification_result
    audit = decision.audit_record
    if result.qualification_request_id != qualification_request.qualification_request_id:
        return "qualification_request_id mismatch"
    if audit.qualification_request_id != qualification_request.qualification_request_id:
        return "audit qualification_request_id mismatch"
    if result.subject_id != subject.subject_id:
        return "result subject mismatch"
    if result.subject_integrity_fingerprint != subject.subject_integrity_fingerprint:
        return "result fingerprint mismatch"
    if result.tenant_id != subject.tenant_id:
        return "result tenant mismatch"
    if result.scope_fingerprint != subject.scope_fingerprint:
        return "result scope fingerprint mismatch"
    if result.subject_id != artifact.artifact_id:
        return "result artifact mismatch"
    if result.tenant_id != artifact.tenant_id:
        return "result artifact tenant mismatch"
    if result.correlation_id != qualification_request.correlation_id:
        return "correlation_id mismatch"
    if result.causation_id != qualification_request.causation_id:
        return "causation_id mismatch"
    evidence = result.evidence
    if evidence is not None:
        if evidence.qualification_request_id != qualification_request.qualification_request_id:
            return "evidence request id mismatch"
        if evidence.subject_id != subject.subject_id:
            return "evidence subject mismatch"
        if evidence.tenant_id != subject.tenant_id:
            return "evidence tenant mismatch"
        if evidence.scope_fingerprint != subject.scope_fingerprint:
            return "evidence scope fingerprint mismatch"
    return None


def validate_execution_bound_qualification_proof(
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
) -> str | None:
    """Mechanical CQ proof at execution boundary — qualification is fact, not permission."""
    decision = handoff.accepted_qualification
    result = decision.qualification_result
    lifecycle = decision.lifecycle_decision
    audit = decision.audit_record
    if result.outcome is not CapabilityQualificationOutcome.QUALIFIED:
        return "qualification outcome not QUALIFIED"
    if lifecycle.outcome is not CapabilityQualificationLifecycleOutcome.ACCEPT:
        return "lifecycle outcome not ACCEPT"
    if handoff.qualification_request_id != result.qualification_request_id:
        return "qualification_request_id mismatch"
    if audit.qualification_request_id != handoff.qualification_request_id:
        return "audit qualification_request_id mismatch"
    subject = handoff.qualification_subject
    artifact = handoff.artifact
    if result.subject_id != subject.subject_id:
        return "result subject mismatch"
    if result.subject_integrity_fingerprint != subject.subject_integrity_fingerprint:
        return "result fingerprint mismatch"
    if result.tenant_id != subject.tenant_id:
        return "result tenant mismatch"
    if result.scope_fingerprint != subject.scope_fingerprint:
        return "result scope fingerprint mismatch"
    if result.subject_id != artifact.artifact_id:
        return "result artifact mismatch"
    if result.tenant_id != artifact.tenant_id:
        return "result artifact tenant mismatch"
    if handoff.tenant_id != artifact.tenant_id:
        return "handoff tenant artifact mismatch"
    if handoff.tenant_id != subject.tenant_id:
        return "handoff tenant subject mismatch"
    if handoff.scope_fingerprint != artifact.scope_fingerprint:
        return "handoff scope fingerprint mismatch"
    if handoff.credential_grant_ref != artifact.scope.credential_grant_ref:
        return "handoff credential grant ref mismatch"
    if result.correlation_id != handoff.correlation_id:
        return "correlation_id mismatch"
    if result.causation_id != handoff.causation_id:
        return "causation_id mismatch"
    evidence = result.evidence
    if evidence is not None:
        if evidence.qualification_request_id != handoff.qualification_request_id:
            return "evidence request id mismatch"
        if evidence.subject_id != subject.subject_id:
            return "evidence subject mismatch"
        if evidence.tenant_id != subject.tenant_id:
            return "evidence tenant mismatch"
        if evidence.scope_fingerprint != subject.scope_fingerprint:
            return "evidence scope fingerprint mismatch"
    return None


def validate_handoff_credential_grant_identity(
    *,
    handoff: ScopedAdaptiveIntegrationExecutionHandoff,
    grant_grant_id: str,
) -> str | None:
    if grant_grant_id != handoff.credential_grant_ref:
        return "credential grant_id does not match handoff credential_grant_ref"
    if grant_grant_id != handoff.artifact.scope.credential_grant_ref:
        return "credential grant_id does not match artifact scope credential_grant_ref"
    return None


def build_scoped_adaptive_integration_execution_handoff(
    *,
    preparation: ScopedAdaptiveIntegrationPreparationResult,
    qualification_request: CapabilityQualificationRequest,
    accepted_qualification: CapabilityQualificationDecision,
    requested_operation: ScopedIntegrationAdaptationOperationId,
    execution_idempotency_key: str,
) -> ScopedAdaptiveIntegrationExecutionHandoff:
    artifact = preparation.artifact
    subject = preparation.qualification_subject
    if artifact is None or subject is None:
        raise ValueError("preparation missing artifact or subject")
    scope = artifact.scope
    if requested_operation not in scope.permitted_operations:
        raise ValueError("requested_operation not in artifact permitted_operations")
    continuity = validate_qualification_decision_continuity(
        qualification_request=qualification_request,
        decision=accepted_qualification,
        artifact=artifact,
        subject=subject,
    )
    if continuity is not None:
        raise ValueError(continuity)
    return ScopedAdaptiveIntegrationExecutionHandoff(
        artifact=artifact,
        qualification_subject=subject,
        accepted_qualification=accepted_qualification,
        qualification_request_id=qualification_request.qualification_request_id,
        requested_operation=requested_operation,
        tenant_id=artifact.tenant_id,
        integration_category=artifact.integration_category,
        provider_id=artifact.provider_id,
        resource_scope=artifact.resource_scope,
        scope_fingerprint=artifact.scope_fingerprint,
        permitted_operations=scope.permitted_operations,
        network_allowlist=scope.network_allowlist,
        credential_grant_ref=scope.credential_grant_ref,
        correlation_id=qualification_request.correlation_id,
        causation_id=qualification_request.causation_id,
        execution_idempotency_key=execution_idempotency_key,
    )


__all__ = [
    "ScopedAdaptiveIntegrationExecutionHandoff",
    "ScopedAdaptiveIntegrationExecutionOutcome",
    "ScopedAdaptiveIntegrationExecutionOutput",
    "ScopedAdaptiveIntegrationExecutionRequest",
    "ScopedAdaptiveIntegrationExecutionResult",
    "ScopedAdaptiveIntegrationExecutionRuntimeEnvelope",
    "ScopedAdaptiveIntegrationWorkerDispatchContext",
    "WorkerScopedAdaptiveIntegrationExecutionCoordinatorPort",
    "build_scoped_adaptive_integration_execution_handoff",
    "validate_execution_bound_qualification_proof",
    "validate_handoff_credential_grant_identity",
    "validate_preparation_artifact_subject_continuity",
    "validate_qualification_decision_continuity",
]
