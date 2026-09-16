# © Artur Czarnecki. All rights reserved.

"""MP-4R7 enterprise integration scenario executor — orchestrates contracts only."""

from __future__ import annotations

from dataclasses import dataclass, replace

from intergrax.collaborative_work.repository import CreateWorkItemCommand
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBindingIdempotencyConflict,
    CollaborativeDecisionBindingScopeMismatch,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_work import MembershipResolutionMode, WorkItemState
from intergrax.contracts.decision_human_review import (
    DecisionHumanReviewMismatchError,
    DecisionHumanReviewOutcome,
    DecisionHumanReviewProvenance,
    decision_human_review_decision,
    validate_human_review_decision_for_proposal,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_lifecycle import DecisionLifecycleStage
from intergrax.contracts.decision_record import (
    DecisionProposalRef,
    decision_lineage_ref,
    validate_decision_artifact_kind,
)
from intergrax.contracts.decision_revision import decision_revision_policy
from intergrax.contracts.decision_lifecycle import transition_decision_lifecycle
from intergrax.contracts.decision_resolution import (
    AuthoritativeResolutionRecord,
    DecisionResolution,
)
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResumeCommand,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    execution_continuation_recovery_handle_for_continuation_id,
    execution_continuation_resolution_command_from_decision_human_review_decision,
)
from intergrax.contracts.functional_evidence import (
    PipelineEvidenceKind,
    PipelineEvidenceScope,
    PipelineOperationStatus,
)
from intergrax.contracts.governed_continuation_correlation import (
    ContinuationReason,
    GovernedContinuationCorrelation,
)
from intergrax.runtime.decision_flow import (
    CanonicalDecisionFlowGate,
    DecisionFlowGateCapabilities,
    DecisionFlowHostAction,
    DecisionFlowIdentitySeed,
    DecisionFlowRequest,
    DecisionFlowScope,
)
from intergrax.runtime.decision_human_review import validate_consumed_human_review_decision
from intergrax.runtime.execution.active_decision_lifecycle_host import (
    bind_active_decision_lifecycle_host,
    reset_active_decision_lifecycle_host,
)
from intergrax.runtime.execution.continuation.composition import (
    reconnect_execution_engine_continuation_dependencies,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import FunctionalDiagnosticAnalyzer
from intergrax.runtime.execution.continuation.persistence import (
    export_durable_continuation_state,
    execution_continuation_state_store_from_durable_export,
)
from intergrax.runtime.execution.continuation.restart_qualification import (
    recover_execution_continuation_process_restart,
)
from intergrax.runtime.execution.decision_lifecycle_host import CanonicalDecisionLifecycleHost
from intergrax.runtime.observability.functional_evidence_recorder import FunctionalEvidenceRecorder
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceError,
)

from testing_support.mp4r7_enterprise_integration.approver import (
    qualification_identity_provider_approver_evidence,
)
from testing_support.mp4r7_enterprise_integration.composition import (
    Mp4R7EnterpriseIntegrationComposition,
)
from testing_support.mp4r7_enterprise_integration.contracts import (
    MP4R7_PROTECTED_OPERATION_ID,
    Mp4R7DecisionIdentitySnapshot,
    Mp4R7DiagnosticsSnapshot,
    Mp4R7EnterpriseIntegrationQualificationResult,
    Mp4R7EvidenceSnapshot,
    Mp4R7ExecutionIdentitySnapshot,
    Mp4R7HumanAuthorityContinuitySnapshot,
    Mp4R7ProtectedOperationError,
    Mp4R7QualificationDisposition,
    Mp4R7ScenarioId,
    Mp4R7WorkBindingSnapshot,
)
from testing_support.mp4r7_enterprise_integration.decision_helpers import (
    Mp4R7PassedVerificationStage,
    Mp4R7RequireHumanGovernanceEvaluator,
    mp4r7_governance_action,
    mp4r7_governance_policy_context,
    mp4r7_governance_spec,
    mp4r7_verification_pipeline,
)
from testing_support.mp4r7_enterprise_integration.diagnostic_spec import (
    mp4r7_protected_operation_diagnostic_specification,
)


@dataclass(frozen=True, slots=True)
class _Payload:
    summary: str


_CONTINUATION_ID = "gcr_mp4r7_enterprise_qualification"
_PAUSE_ID = "pause_mp4r7"
_OPERATION_ID = "op_mp4r7"
_SIDE_EFFECT_SCOPE = "scope_mp4r7"
_RESOLVED_AT = "2026-09-16T12:30:00Z"


class Mp4R7EnterpriseIntegrationScenarioExecutor:
    """Calls platform contracts in order — owns no domain truth."""

    def __init__(self, composition: Mp4R7EnterpriseIntegrationComposition) -> None:
        self._composition = composition

    def _execution_lineage(self) -> DecisionExecutionLineage:
        composition = self._composition
        return DecisionExecutionLineage(
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
        )

    def _continuation_identity(self) -> ExecutionContinuationIdentity:
        composition = self._composition
        return ExecutionContinuationIdentity(
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
        )

    def _governed_correlation(self) -> GovernedContinuationCorrelation:
        composition = self._composition
        return GovernedContinuationCorrelation(
            continuation_request_id=_CONTINUATION_ID,
            reason=ContinuationReason.COMPLIANCE,
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
            operation_id=_OPERATION_ID,
            side_effect_scope_id=_SIDE_EFFECT_SCOPE,
        )

    def _pause_request(self, human_request_id: str) -> ExecutionPauseRequest:
        return ExecutionPauseRequest(
            identity=self._continuation_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.COMPLIANCE,
            governed_correlation=self._governed_correlation(),
            pause_id=_PAUSE_ID,
            human_request_id=human_request_id,
        )

    def _human_authority_snapshot(
        self,
        *,
        phase: str,
        human_request_id: str,
        decision,
    ) -> Mp4R7HumanAuthorityContinuitySnapshot:
        proposal = decision.proposal_ref
        return Mp4R7HumanAuthorityContinuitySnapshot(
            phase=phase,
            human_request_id=human_request_id,
            approver_user_id=decision.approver.user_id,
            approver_tenant_id=decision.approver.tenant_id,
            proposal_decision_id=str(proposal.identity.decision_id),
            proposal_version=str(proposal.identity.version),
        )

    def _apply_human_review_to_continuation(
        self,
        waiting: PendingExecutionContinuation,
        decision,
    ) -> PendingExecutionContinuation:
        command = execution_continuation_resolution_command_from_decision_human_review_decision(
            waiting,
            decision,
            resolved_at=_RESOLVED_AT,
        )
        return self._composition.continuation_port.apply_resolution(command)

    def _finalize_decision_lifecycle_after_human_review(
        self,
        flow,
        decision,
    ) -> tuple[DecisionLifecycleStage, ...]:
        lifecycle = flow.lifecycle_state
        if decision.outcome is DecisionHumanReviewOutcome.APPROVED:
            transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
            return (
                lifecycle.stage,
                DecisionLifecycleStage.FINALIZATION,
                DecisionLifecycleStage.TERMINAL,
            )
        if decision.outcome is DecisionHumanReviewOutcome.REJECTED:
            identity = decision.proposal_ref.identity
            AuthoritativeResolutionRecord(
                identity=identity,
                resolution=DecisionResolution.REJECTED,
            )
            transition_decision_lifecycle(lifecycle, DecisionLifecycleStage.TERMINAL)
            return (lifecycle.stage, DecisionLifecycleStage.TERMINAL)
        raise AssertionError("unsupported human review outcome for decision finalization")

    def _resume_command(self, *, expected_revision: int) -> ExecutionContinuationResumeCommand:
        return ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=self._continuation_identity(),
            expected_revision=expected_revision,
        )

    def _identity_snapshot(self, phase: str) -> Mp4R7ExecutionIdentitySnapshot:
        composition = self._composition
        return Mp4R7ExecutionIdentitySnapshot(
            phase=phase,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
        )

    def _seed_work_item(self) -> None:
        composition = self._composition
        now = composition.clock()
        composition.work_item_repository.create(
            CreateWorkItemCommand(
                tenant_id=composition.tenant_id,
                workspace_id=composition.workspace_id,
                work_item_id=composition.work_item_id,
                created_by_principal_id=composition.acting_principal_id,
                state=WorkItemState.OPEN,
                created_at=now,
                updated_at=now,
                title="MP-4R7 enterprise integration qualification",
            ),
        )

    async def _run_decision_flow_require_human(self):
        composition = self._composition
        evaluator = Mp4R7RequireHumanGovernanceEvaluator(
            action=mp4r7_governance_action(),
            policy_context=mp4r7_governance_policy_context(),
        )
        gate = CanonicalDecisionFlowGate(
            capabilities=DecisionFlowGateCapabilities(
                verification_pipeline=mp4r7_verification_pipeline(
                    Mp4R7PassedVerificationStage(),
                ),
                revision_policy=decision_revision_policy(max_revisions=0),
                scopes=frozenset({DecisionFlowScope.UAEP_STEP}),
                governance_spec=mp4r7_governance_spec(evaluator),
                human_review_port=composition.human_review_port,
            ),
        )
        lineage = self._execution_lineage()
        token = bind_active_decision_lifecycle_host(CanonicalDecisionLifecycleHost())
        try:
            return await gate.evaluate(
                DecisionFlowRequest(
                    identity_seed=DecisionFlowIdentitySeed(
                        scope=DecisionScope(namespace="mp4r7", subject="enterprise-integration"),
                        tenant_id=composition.tenant_id,
                        execution=lineage,
                        decision_id=mint_decision_id(),
                    ),
                    artifact_kind=validate_decision_artifact_kind("mp4r7.enterprise.payload"),
                    payload=_Payload(summary="consequential enterprise change"),
                    flow_scope=DecisionFlowScope.UAEP_STEP,
                ),
            )
        finally:
            reset_active_decision_lifecycle_host(token)

    def _create_binding(
        self,
        proposal_ref: DecisionProposalRef,
        *,
        idempotency_key: str = "mp4r7-binding-idem",
    ):
        composition = self._composition
        return composition.binding_application.create_binding(
            CreateCollaborativeDecisionBindingRequest(
                tenant_id=composition.tenant_id,
                workspace_id=composition.workspace_id,
                work_item_id=composition.work_item_id,
                decision_proposal=proposal_ref,
                acting_principal_id=composition.acting_principal_id,
                idempotency_key=idempotency_key,
                membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            ),
        )

    def _drive_execution_to_waiting(self, human_request_id: str) -> PendingExecutionContinuation:
        composition = self._composition
        port = composition.continuation_port
        driver = composition.continuation_dependencies.lifecycle_driver
        port.request_pause(self._pause_request(human_request_id))
        driver.record_execution_reached_safe_pause(
            _CONTINUATION_ID,
            execution_pause_established=True,
        )
        waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
        if waiting.lifecycle_state is not ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN:
            raise AssertionError("execution must reach WAITING_FOR_HUMAN via canonical owner")
        return waiting

    def _resume_after_human_review(
        self,
        waiting: PendingExecutionContinuation,
        decision,
    ) -> PendingExecutionContinuation:
        port = self._composition.continuation_port
        authorized = self._apply_human_review_to_continuation(waiting, decision)
        if authorized.lifecycle_state is not ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            raise AssertionError("two-phase resume requires RESUME_AUTHORIZED")
        return port.resume(self._resume_command(expected_revision=authorized.revision))

    def _record_protected_operation(
        self,
        *,
        persistence: FunctionalEvidencePersistence | None = None,
        fail_append: bool = False,
    ) -> tuple[Mp4R7EvidenceSnapshot, Exception | None]:
        composition = self._composition
        store = persistence or composition.evidence_persistence
        recorder = FunctionalEvidenceRecorder(
            persistence=store,
            producer_component="testing_support.mp4r7",
        )
        scope = PipelineEvidenceScope(
            tenant_id=composition.tenant_id,
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
        )
        primary_error: Exception | None = None
        if fail_append:
            primary_error = Mp4R7ProtectedOperationError("protected operation failed")
            try:
                raise primary_error
            except Mp4R7ProtectedOperationError as primary:
                class _FailingPersistence(FunctionalEvidencePersistence):
                    def __init__(self, inner: FunctionalEvidencePersistence) -> None:
                        self._inner = inner

                    def append(self, evidence):  # type: ignore[no-untyped-def]
                        raise FunctionalEvidencePersistenceError(
                            "mp4r7 simulated evidence sink failure",
                        )

                    def query_evidence(self, request):  # type: ignore[no-untyped-def]
                        return self._inner.query_evidence(request)

                store = _FailingPersistence(composition.evidence_persistence)
                recorder = FunctionalEvidenceRecorder(
                    persistence=store,
                    producer_component="testing_support.mp4r7",
                )
                try:
                    recorder.record_operation_outcome(
                        scope=scope,
                        operation_id=MP4R7_PROTECTED_OPERATION_ID,
                        operation_name="mp4r7 protected side effect",
                        status=PipelineOperationStatus.FAILED,
                    )
                except FunctionalEvidencePersistenceError:
                    pass
                raise primary
        evidence = recorder.record_operation_outcome(
            scope=scope,
            operation_id=MP4R7_PROTECTED_OPERATION_ID,
            operation_name="mp4r7 protected side effect",
            status=PipelineOperationStatus.SUCCEEDED,
        )
        if evidence is None:
            raise AssertionError("operation_outcome evidence must be persisted")
        return (
            Mp4R7EvidenceSnapshot(
                evidence_id=evidence.evidence_id,
                kind=evidence.kind.value,
                operation_id=MP4R7_PROTECTED_OPERATION_ID,
                status=PipelineOperationStatus.SUCCEEDED,
            ),
            primary_error,
        )

    def _diagnostics_snapshot(self) -> Mp4R7DiagnosticsSnapshot:
        composition = self._composition
        analyzer = FunctionalDiagnosticAnalyzer(composition.evidence_persistence)
        analysis = analyzer.analyze(
            tenant_id=composition.tenant_id,
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            specification=mp4r7_protected_operation_diagnostic_specification(),
        )
        check = analysis.check_results[0]
        return Mp4R7DiagnosticsSnapshot(
            specification_id=str(analysis.specification_id),
            first_failure_check_id=(
                str(analysis.first_proven_failure)
                if analysis.first_proven_failure is not None
                else None
            ),
            operation_outcome_check_status=check.status.value,
        )

    async def run_success(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        if flow.host_action is not DecisionFlowHostAction.PENDING_HUMAN:
            raise AssertionError("governance must require human review")
        if flow.authority_reason != "decision_governance_human_review_pending":
            raise AssertionError("governance must surface REQUIRE_HUMAN via canonical reason")
        pending = flow.human_review_pending
        if pending is None:
            raise AssertionError("human review pending required")
        proposal_ref = pending.request.proposal_ref
        human_request_id = str(pending.request.request_id)
        binding = self._create_binding(proposal_ref)
        assert binding.binding_id
        waiting = self._drive_execution_to_waiting(human_request_id)
        decision = decision_human_review_decision(
            request=pending.request,
            outcome=DecisionHumanReviewOutcome.APPROVED,
            approver=qualification_identity_provider_approver_evidence(
                tenant_id=composition.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_mp4r7_success",
                human_request_id=human_request_id,
            ),
        )
        validate_consumed_human_review_decision(
            request=pending.request,
            decision=decision,
            target_proposal_ref=proposal_ref,
        )
        decision_stages = self._finalize_decision_lifecycle_after_human_review(flow, decision)
        resumed = self._resume_after_human_review(waiting, decision)
        if resumed.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED:
            raise AssertionError("execution must reach RESUMED")
        evidence_snapshot, _ = self._record_protected_operation()
        diagnostics = self._diagnostics_snapshot()
        authority_snapshots = (
            self._human_authority_snapshot(
                phase="decision_human_review",
                human_request_id=human_request_id,
                decision=decision,
            ),
            self._human_authority_snapshot(
                phase="continuation_resolution",
                human_request_id=human_request_id,
                decision=decision,
            ),
        )
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.SUCCESS,
            disposition=Mp4R7QualificationDisposition.QUALIFIED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=proposal_ref,
            governance_required_human=True,
            human_outcome=DecisionHumanReviewOutcome.APPROVED,
            continuation_result_state=resumed.lifecycle_state,
            protected_operation_completed=True,
            execution_identities=(
                self._identity_snapshot("before_hitl"),
                self._identity_snapshot("paused"),
                self._identity_snapshot("resumed"),
                self._identity_snapshot("completed"),
            ),
            decision_identities=(
                Mp4R7DecisionIdentitySnapshot(
                    phase="governance_hitl",
                    decision_id=str(proposal_ref.identity.decision_id),
                    decision_version=str(proposal_ref.identity.version),
                    proposal_ref=proposal_ref,
                ),
            ),
            work_binding=Mp4R7WorkBindingSnapshot(
                work_item_id=composition.work_item_id,
                artifact_version_ref=None,
                decision_proposal_ref=proposal_ref,
            ),
            evidence_records=(evidence_snapshot,),
            diagnostics=diagnostics,
            decision_lifecycle_stages_observed=decision_stages,
            pause_id=waiting.pause_id,
            human_request_id=human_request_id,
            continuation_id=_CONTINUATION_ID,
            human_authority_continuity=authority_snapshots,
            decision_final_stage=DecisionLifecycleStage.TERMINAL,
        )

    async def run_human_reject(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        proposal_ref = pending.request.proposal_ref
        human_request_id = str(pending.request.request_id)
        self._create_binding(proposal_ref)
        waiting = self._drive_execution_to_waiting(human_request_id)
        decision = decision_human_review_decision(
            request=pending.request,
            outcome=DecisionHumanReviewOutcome.REJECTED,
            approver=qualification_identity_provider_approver_evidence(
                tenant_id=composition.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_mp4r7_reject",
                human_request_id=human_request_id,
            ),
        )
        validate_consumed_human_review_decision(
            request=pending.request,
            decision=decision,
            target_proposal_ref=proposal_ref,
        )
        self._finalize_decision_lifecycle_after_human_review(flow, decision)
        resolved = self._apply_human_review_to_continuation(waiting, decision)
        if resolved.lifecycle_state is not ExecutionContinuationLifecycleState.REJECTED:
            raise AssertionError("human reject must terminal REJECTED continuation state")
        try:
            self._composition.continuation_port.resume(
                self._resume_command(expected_revision=resolved.revision),
            )
        except ExecutionContinuationError:
            pass
        else:
            raise AssertionError("resume must fail closed after human reject")
        current = self._composition.continuation_port.get_pending(
            ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID),
        )
        assert current.lifecycle_state is not ExecutionContinuationLifecycleState.RESUMED
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.HUMAN_REJECT,
            disposition=Mp4R7QualificationDisposition.QUALIFIED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=proposal_ref,
            governance_required_human=True,
            human_outcome=DecisionHumanReviewOutcome.REJECTED,
            continuation_result_state=current.lifecycle_state,
            protected_operation_completed=False,
            execution_identities=(
                self._identity_snapshot("before_hitl"),
                self._identity_snapshot("paused"),
            ),
            decision_identities=(
                Mp4R7DecisionIdentitySnapshot(
                    phase="reject",
                    decision_id=str(proposal_ref.identity.decision_id),
                    decision_version=str(proposal_ref.identity.version),
                    proposal_ref=proposal_ref,
                ),
            ),
            work_binding=Mp4R7WorkBindingSnapshot(
                work_item_id=composition.work_item_id,
                artifact_version_ref=None,
                decision_proposal_ref=proposal_ref,
            ),
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(flow.lifecycle_state.stage,),
            pause_id=waiting.pause_id,
            human_request_id=human_request_id,
            continuation_id=_CONTINUATION_ID,
            decision_final_stage=DecisionLifecycleStage.TERMINAL,
        )

    async def run_stale_proposal(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        stale_ref = pending.request.proposal_ref
        current_identity = stale_ref.identity
        bumped = DecisionProposalRef(
            identity=replace(
                current_identity,
                version=next_decision_version(current_identity.version),
            ),
            lineage_ref=decision_lineage_ref(next_decision_version(current_identity.version)),
        )
        decision = decision_human_review_decision(
            request=pending.request,
            outcome=DecisionHumanReviewOutcome.APPROVED,
            approver=qualification_identity_provider_approver_evidence(
                tenant_id=composition.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_mp4r7_stale",
                human_request_id=str(pending.request.request_id),
            ),
        )
        try:
            validate_human_review_decision_for_proposal(
                decision=decision,
                proposal_ref=bumped,
            )
            raise AssertionError("stale proposal must fail closed")
        except DecisionHumanReviewMismatchError:
            pass
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.STALE_PROPOSAL,
            disposition=Mp4R7QualificationDisposition.FAIL_CLOSED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=stale_ref,
            governance_required_human=True,
            human_outcome=None,
            continuation_result_state=None,
            protected_operation_completed=False,
            execution_identities=(self._identity_snapshot("stale_check"),),
            decision_identities=(
                Mp4R7DecisionIdentitySnapshot(
                    phase="stale",
                    decision_id=str(stale_ref.identity.decision_id),
                    decision_version=str(stale_ref.identity.version),
                    proposal_ref=stale_ref,
                ),
            ),
            work_binding=None,
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(flow.lifecycle_state.stage,),
            pause_id=None,
            human_request_id=str(pending.request.request_id),
            continuation_id=None,
            primary_error_code="DecisionHumanReviewMismatchError",
        )

    async def run_cross_tenant(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        proposal_ref = pending.request.proposal_ref
        mismatched_proposal = DecisionProposalRef(
            identity=replace(proposal_ref.identity, tenant_id="tenant-other"),
            lineage_ref=proposal_ref.lineage_ref,
        )
        try:
            composition.binding_application.create_binding(
                CreateCollaborativeDecisionBindingRequest(
                    tenant_id=composition.tenant_id,
                    workspace_id=composition.workspace_id,
                    work_item_id=composition.work_item_id,
                    decision_proposal=mismatched_proposal,
                    acting_principal_id=composition.acting_principal_id,
                    idempotency_key="cross-tenant-mp4r7",
                    membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
                ),
            )
            raise AssertionError("cross-tenant binding must fail closed")
        except CollaborativeDecisionBindingScopeMismatch:
            pass
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.CROSS_TENANT,
            disposition=Mp4R7QualificationDisposition.FAIL_CLOSED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=proposal_ref,
            governance_required_human=True,
            human_outcome=None,
            continuation_result_state=None,
            protected_operation_completed=False,
            execution_identities=(self._identity_snapshot("cross_tenant"),),
            decision_identities=(
                Mp4R7DecisionIdentitySnapshot(
                    phase="cross_tenant",
                    decision_id=str(proposal_ref.identity.decision_id),
                    decision_version=str(proposal_ref.identity.version),
                    proposal_ref=proposal_ref,
                ),
            ),
            work_binding=None,
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(flow.lifecycle_state.stage,),
            pause_id=None,
            human_request_id=str(pending.request.request_id),
            continuation_id=None,
            primary_error_code="DecisionHumanReviewMismatchError",
        )

    async def run_evidence_failure(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        secondary_error_code: str | None = None
        primary_error_code: str
        try:
            self._record_protected_operation(fail_append=True)
        except Mp4R7ProtectedOperationError as primary:
            secondary_error_code = "FunctionalEvidencePersistenceError"
            primary_error_code = type(primary).__name__
        else:
            raise AssertionError("protected operation must fail before evidence emission")
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.EVIDENCE_FAILURE,
            disposition=Mp4R7QualificationDisposition.QUALIFIED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=DecisionProposalRef(
                identity=mint_decision_placeholder_identity(composition),
                lineage_ref=decision_lineage_ref(initial_decision_version()),
            ),
            governance_required_human=False,
            human_outcome=None,
            continuation_result_state=None,
            protected_operation_completed=False,
            execution_identities=(self._identity_snapshot("evidence_failure"),),
            decision_identities=(),
            work_binding=None,
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(),
            pause_id=None,
            human_request_id=None,
            continuation_id=None,
            primary_error_code=primary_error_code,
            secondary_evidence_error_code=secondary_error_code,
        )

    async def run_binding_idempotency(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        proposal_ref = pending.request.proposal_ref
        first = self._create_binding(proposal_ref, idempotency_key="idem-mp4r7")
        second = self._create_binding(proposal_ref, idempotency_key="idem-mp4r7")
        assert first.binding_id == second.binding_id
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.BINDING_IDEMPOTENCY,
            disposition=Mp4R7QualificationDisposition.QUALIFIED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=proposal_ref,
            governance_required_human=True,
            human_outcome=None,
            continuation_result_state=None,
            protected_operation_completed=False,
            execution_identities=(self._identity_snapshot("binding_idempotent"),),
            decision_identities=(),
            work_binding=Mp4R7WorkBindingSnapshot(
                work_item_id=composition.work_item_id,
                artifact_version_ref=None,
                decision_proposal_ref=proposal_ref,
            ),
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(flow.lifecycle_state.stage,),
            pause_id=None,
            human_request_id=None,
            continuation_id=None,
        )

    async def run_binding_conflict(self) -> None:
        composition = self._composition
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        proposal_ref = pending.request.proposal_ref
        self._create_binding(proposal_ref, idempotency_key="conflict-mp4r7")
        other_identity = replace(proposal_ref.identity, decision_id=mint_decision_id())
        other_ref = DecisionProposalRef(
            identity=other_identity,
            lineage_ref=decision_lineage_ref(other_identity.version),
        )
        try:
            self._create_binding(other_ref, idempotency_key="conflict-mp4r7")
        except CollaborativeDecisionBindingIdempotencyConflict:
            return
        raise AssertionError("semantic idempotency conflict expected")

    async def run_process_restart(self) -> Mp4R7EnterpriseIntegrationQualificationResult:
        composition = self._composition
        if composition.continuation_backing is None:
            return Mp4R7EnterpriseIntegrationQualificationResult(
                scenario_id=Mp4R7ScenarioId.PROCESS_RESTART,
                disposition=Mp4R7QualificationDisposition.BLOCKED,
                tenant_id=composition.tenant_id,
                workspace_id=composition.workspace_id,
                task_id=composition.task_id,
                work_item_id=composition.work_item_id,
                decision_proposal_ref=DecisionProposalRef(
                    identity=mint_decision_placeholder_identity(composition),
                    lineage_ref=decision_lineage_ref(initial_decision_version()),
                ),
                governance_required_human=False,
                human_outcome=None,
                continuation_result_state=None,
                protected_operation_completed=False,
                execution_identities=(self._identity_snapshot("restart_blocked"),),
                decision_identities=(),
                work_binding=None,
                evidence_records=(),
                diagnostics=None,
                decision_lifecycle_stages_observed=(),
                pause_id=None,
                human_request_id=None,
                continuation_id=_CONTINUATION_ID,
                primary_error_code="NON_DURABLE_CONTINUATION_STORE",
            )
        self._seed_work_item()
        flow = await self._run_decision_flow_require_human()
        pending = flow.human_review_pending
        assert pending is not None
        human_request_id = str(pending.request.request_id)
        proposal_ref = pending.request.proposal_ref
        self._create_binding(proposal_ref)
        waiting = self._drive_execution_to_waiting(human_request_id)
        decision = decision_human_review_decision(
            request=pending.request,
            outcome=DecisionHumanReviewOutcome.APPROVED,
            approver=qualification_identity_provider_approver_evidence(
                tenant_id=composition.tenant_id,
            ),
            provenance=DecisionHumanReviewProvenance(
                human_record_id="hdec_mp4r7_restart",
                human_request_id=human_request_id,
            ),
        )
        validate_consumed_human_review_decision(
            request=pending.request,
            decision=decision,
            target_proposal_ref=proposal_ref,
        )
        authorized = self._apply_human_review_to_continuation(waiting, decision)
        if authorized.lifecycle_state is not ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            raise AssertionError("restart qualification requires RESUME_AUTHORIZED before export")
        export = export_durable_continuation_state(composition.continuation_backing)
        restored_store = execution_continuation_state_store_from_durable_export(export)
        reconnected = reconnect_execution_engine_continuation_dependencies(
            state_store=restored_store,
        )
        handle = execution_continuation_recovery_handle_for_continuation_id(_CONTINUATION_ID)
        qual = recover_execution_continuation_process_restart(
            store=restored_store,
            recovery_handle=handle,
            authority=ParentExecutionAuthority.unrestricted_root(),
            tenant_id=composition.tenant_id,
            task_id_consistency=composition.task_id,
            expected_identity=self._continuation_identity(),
        )
        port = reconnected.continuation
        if qual.current_episode.lifecycle_state is not ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
            raise AssertionError("restored continuation must remain RESUME_AUTHORIZED")
        resumed = port.resume(
            self._resume_command(expected_revision=qual.current_episode.revision),
        )
        return Mp4R7EnterpriseIntegrationQualificationResult(
            scenario_id=Mp4R7ScenarioId.PROCESS_RESTART,
            disposition=Mp4R7QualificationDisposition.QUALIFIED,
            tenant_id=composition.tenant_id,
            workspace_id=composition.workspace_id,
            task_id=composition.task_id,
            work_item_id=composition.work_item_id,
            decision_proposal_ref=proposal_ref,
            governance_required_human=True,
            human_outcome=DecisionHumanReviewOutcome.APPROVED,
            continuation_result_state=resumed.lifecycle_state,
            protected_operation_completed=False,
            execution_identities=(
                self._identity_snapshot("paused"),
                self._identity_snapshot("restored"),
                self._identity_snapshot("resumed"),
            ),
            decision_identities=(),
            work_binding=None,
            evidence_records=(),
            diagnostics=None,
            decision_lifecycle_stages_observed=(),
            pause_id=qual.current_episode.pause_id,
            human_request_id=human_request_id,
            continuation_id=_CONTINUATION_ID,
            human_authority_continuity=(
                self._human_authority_snapshot(
                    phase="decision_human_review",
                    human_request_id=human_request_id,
                    decision=decision,
                ),
            ),
            decision_final_stage=DecisionLifecycleStage.TERMINAL,
        )


def mint_decision_placeholder_identity(
    composition: Mp4R7EnterpriseIntegrationComposition,
):
    from intergrax.contracts.decision_identity import DecisionIdentity

    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="mp4r7", subject="placeholder"),
        tenant_id=composition.tenant_id,
        execution=DecisionExecutionLineage(
            task_id=composition.task_id,
            run_id=composition.run_id,
            attempt_id=composition.attempt_id,
            execution_id=composition.execution_id,
        ),
    )


__all__ = ["Mp4R7EnterpriseIntegrationScenarioExecutor"]
