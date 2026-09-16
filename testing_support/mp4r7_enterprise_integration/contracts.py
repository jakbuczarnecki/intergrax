# © Artur Czarnecki. All rights reserved.

"""Typed MP-4R7 enterprise integration qualification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.decision_authorization import DecisionGovernanceDisposition
from intergrax.contracts.decision_human_review import DecisionHumanReviewOutcome
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import DecisionProposalRef
from intergrax.contracts.execution_continuation import ExecutionContinuationLifecycleState
from intergrax.contracts.execution_identity import AttemptId, EventId, ExecutionId, RunId, TaskId
from intergrax.contracts.functional_evidence import PipelineOperationStatus
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticAnalysis


class Mp4R7ScenarioId(StrEnum):
    SUCCESS = "mp4r7.success"
    HUMAN_REJECT = "mp4r7.human_reject"
    STALE_PROPOSAL = "mp4r7.stale_proposal"
    CROSS_TENANT = "mp4r7.cross_tenant"
    EVIDENCE_FAILURE = "mp4r7.evidence_failure"
    BINDING_IDEMPOTENCY = "mp4r7.binding_idempotency"
    HUMAN_REPLAY = "mp4r7.human_replay"
    PROCESS_RESTART = "mp4r7.process_restart"
    GOVERNANCE_DENY = "mp4r7.governance_deny"
    STALE_EXECUTION_POLICY = "mp4r7.stale_execution_policy"


class Mp4R7QualificationDisposition(StrEnum):
    QUALIFIED = "qualified"
    FAIL_CLOSED = "fail_closed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class Mp4R7ExecutionIdentitySnapshot:
    phase: str
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId


@dataclass(frozen=True, slots=True)
class Mp4R7DecisionIdentitySnapshot:
    phase: str
    decision_id: str
    decision_version: str
    proposal_ref: DecisionProposalRef


@dataclass(frozen=True, slots=True)
class Mp4R7WorkBindingSnapshot:
    work_item_id: str
    artifact_version_ref: str | None
    decision_proposal_ref: DecisionProposalRef


@dataclass(frozen=True, slots=True)
class Mp4R7EvidenceSnapshot:
    evidence_id: EventId
    kind: str
    operation_id: str
    status: PipelineOperationStatus | None


@dataclass(frozen=True, slots=True)
class Mp4R7HumanAuthorityContinuitySnapshot:
    phase: str
    human_request_id: str
    approver_user_id: str
    approver_tenant_id: str
    proposal_decision_id: str
    proposal_version: str


class Mp4R7ProtectedOperationError(RuntimeError):
    """Qualification-only protected side-effect failure (primary domain error)."""


@dataclass(frozen=True, slots=True)
class Mp4R7DiagnosticsSnapshot:
    specification_id: str
    first_failure_check_id: str | None
    operation_outcome_check_status: str


@dataclass(frozen=True, slots=True)
class Mp4R7EnterpriseIntegrationQualificationResult:
    scenario_id: Mp4R7ScenarioId
    disposition: Mp4R7QualificationDisposition
    tenant_id: str
    workspace_id: str
    task_id: TaskId
    work_item_id: str
    decision_proposal_ref: DecisionProposalRef
    governance_required_human: bool
    human_outcome: DecisionHumanReviewOutcome | None
    continuation_result_state: ExecutionContinuationLifecycleState | None
    protected_operation_completed: bool
    execution_identities: tuple[Mp4R7ExecutionIdentitySnapshot, ...]
    decision_identities: tuple[Mp4R7DecisionIdentitySnapshot, ...]
    work_binding: Mp4R7WorkBindingSnapshot | None
    evidence_records: tuple[Mp4R7EvidenceSnapshot, ...]
    diagnostics: Mp4R7DiagnosticsSnapshot | None
    decision_lifecycle_stages_observed: tuple[DecisionLifecycleStage, ...]
    pause_id: str | None
    human_request_id: str | None
    continuation_id: str | None
    primary_error_code: str | None = None
    human_authority_continuity: tuple[Mp4R7HumanAuthorityContinuitySnapshot, ...] = ()
    decision_final_stage: DecisionLifecycleStage | None = None
    secondary_evidence_error_code: str | None = None
    post_human_governance_disposition: DecisionGovernanceDisposition | None = None
    execution_authorization_present: bool = False
    execution_authorization_validated: bool = False


MP4R7_PROTECTED_OPERATION_ID = "mp4r7.enterprise.protected_side_effect"
MP4R7_SCENARIO_SEED = "mp4r7-enterprise-integration"


__all__ = [
    "MP4R7_PROTECTED_OPERATION_ID",
    "MP4R7_SCENARIO_SEED",
    "Mp4R7DecisionIdentitySnapshot",
    "Mp4R7DiagnosticsSnapshot",
    "Mp4R7EnterpriseIntegrationQualificationResult",
    "Mp4R7EvidenceSnapshot",
    "Mp4R7HumanAuthorityContinuitySnapshot",
    "Mp4R7ProtectedOperationError",
    "Mp4R7ExecutionIdentitySnapshot",
    "Mp4R7QualificationDisposition",
    "Mp4R7ScenarioId",
    "Mp4R7WorkBindingSnapshot",
]
