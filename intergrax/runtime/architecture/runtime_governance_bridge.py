# © Artur Czarnecki. All rights reserved.

"""Runtime wiring for Phase V architecture governance contracts."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.control_plane_mutation import ControlPlaneMutationAuthorizationEvidence
from intergrax.runtime.adaptive.adaptation_executor import (
    AdaptationExecutor,
    ApplyProfileResult,
    RollbackProfileResult,
)
from intergrax.runtime.adaptive.adaptation_models import AdaptationProposalPackage
from intergrax.runtime.adaptive.control_plane_governance import (
    AhiGovernanceBlockedError,
    AhiTenantScopeResolver,
    authorize_scoped_ahi_control_plane_mutation,
    build_apply_profile_mutation_request,
    build_rollback_profile_mutation_request,
    DirectAhiTenantScopeResolver,
    enforce_ahi_authorization_result,
    validate_ahi_principal_tenant_authority,
)
from intergrax.runtime.adaptive.contracts import ProfileArtifactType
from intergrax.runtime.adaptive.profile_pointer_store import ProfileActivePointerStore
from intergrax.runtime.architecture.adaptive_governance import (
    AdaptiveGovernanceReport,
    AdaptiveLoopProposal,
    evaluate_adaptive_governance,
    evaluate_bounded_adaptive_loop,
)
from intergrax.runtime.architecture.graph_provenance import GraphTraceFieldBundle, build_graph_provenance_trace
from intergrax.runtime.architecture.graph_rag import GraphRagEdge, GraphRagNode
from intergrax.runtime.architecture.multi_agent_coordination import (
    PatternSelectionMatrixReport,
    PlanningConstraints,
    select_coordination_pattern,
)
from intergrax.runtime.architecture.online_evaluation import OnlineEvaluationObservation, record_shadow_observation
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.governance.control_plane_mutation_authorization import (
    ControlPlaneMutationAuthorizationBoundary,
)


class RuntimeGovernanceTraceMetadata(BaseModel):
    coordination_pattern: str = ""
    adaptive_governance_passed: bool = True
    graph_trace_id: str = ""
    candidate_profile_version_id: str = ""
    reasons: list[str] = Field(default_factory=list)


class GovernedApplyProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    apply_result: ApplyProfileResult
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence


class GovernedRollbackProfileResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rollback_result: RollbackProfileResult
    authorization_evidence: ControlPlaneMutationAuthorizationEvidence


class RuntimeArchitectureGovernanceBridge:
    """Typed bridge used by Nexus runtime to emit architecture governance metadata."""

    def __init__(
        self,
        *,
        evaluation_registry: OnlineEvaluationRegistry | None = None,
        mutation_authorization_boundary: ControlPlaneMutationAuthorizationBoundary | None = None,
        tenant_scope_resolver: AhiTenantScopeResolver | None = None,
    ) -> None:
        self._evaluation_registry = evaluation_registry
        self._mutation_authorization_boundary = mutation_authorization_boundary
        self._tenant_scope_resolver = tenant_scope_resolver or DirectAhiTenantScopeResolver()

    def select_coordination_pattern(
        self,
        constraints: PlanningConstraints,
    ) -> PatternSelectionMatrixReport:
        return select_coordination_pattern(constraints=constraints)

    def evaluate_adaptive_proposal(self, proposal: AdaptiveLoopProposal) -> AdaptiveGovernanceReport:
        return evaluate_adaptive_governance([proposal])

    def build_graph_trace_bundle(
        self,
        *,
        trace_id: str,
        graph_id: str,
        nodes: list[GraphRagNode],
        edges: list[GraphRagEdge],
        target_node_id: str,
    ) -> GraphTraceFieldBundle:
        return build_graph_provenance_trace(
            trace_id=trace_id,
            graph_id=graph_id,
            nodes=nodes,
            edges=edges,
            target_node_id=target_node_id,
        )

    def build_trace_metadata(
        self,
        *,
        constraints: PlanningConstraints | None = None,
        adaptive_proposal: AdaptiveLoopProposal | None = None,
        candidate_profile_version_id: str | None = None,
    ) -> RuntimeGovernanceTraceMetadata:
        reasons: list[str] = []
        pattern_name = ""
        if constraints is not None:
            selection = self.select_coordination_pattern(constraints)
            pattern_name = selection.decision.selected_pattern.value
        adaptive_passed = True
        if adaptive_proposal is not None:
            gate = evaluate_bounded_adaptive_loop(adaptive_proposal)
            adaptive_passed = gate.passed
            if gate.reasons:
                reasons.extend(gate.reasons)
        return RuntimeGovernanceTraceMetadata(
            coordination_pattern=pattern_name,
            adaptive_governance_passed=adaptive_passed,
            candidate_profile_version_id=candidate_profile_version_id or "",
            reasons=reasons,
        )

    def record_shadow_run_evaluation(
        self,
        *,
        run_id: str,
        agent_id: str,
        scenario_id: str,
        passed: bool,
        score: float,
        candidate_profile_version_id: str | None = None,
    ) -> OnlineEvaluationObservation:
        """Append a shadow-mode harness evaluation observation (W-OPS.11, W-ADAPT-3.4)."""
        return record_shadow_observation(
            run_id=run_id,
            agent_id=agent_id,
            scenario_id=scenario_id,
            passed=passed,
            score=score,
            registry=self._evaluation_registry,
            candidate_profile_version_id=candidate_profile_version_id,
        )

    def submit_proposal(self, package: AdaptationProposalPackage) -> str:
        """Register a governed proposal for ops/audit review (W-ADAPT-4.8)."""
        if not package.passed_all_gates:
            raise ValueError("Cannot submit a proposal that failed governance gates")
        gate = evaluate_bounded_adaptive_loop(package.candidate.proposal)
        if not gate.passed:
            raise ValueError(f"Adaptive proposal failed envelope gate: {gate.reasons}")
        return package.proposal_id

    def apply_approved(
        self,
        package: AdaptationProposalPackage,
        *,
        executor: AdaptationExecutor,
        pointer_store: ProfileActivePointerStore,
        principal: RequestIdentity,
        mutation_id: str,
        tenant_id: str,
        task_class: str,
        version_id: str,
    ) -> GovernedApplyProfileResult:
        """Apply a previously gated proposal through canonical control-plane authorization."""
        self.submit_proposal(package)
        draft = package.candidate.profile_draft
        if draft is None:
            raise ValueError("Apply requires a ProfileVersionDraft")
        artifact_type = draft.artifact_type

        validate_ahi_principal_tenant_authority(
            principal=principal,
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            operation="apply_profile",
        )

        pointer = pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        current_active_version_id = pointer.active_version_id if pointer is not None else None

        boundary = self._require_mutation_authorization_boundary()
        request = build_apply_profile_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            task_class=task_class,
            mutation_id=mutation_id,
            artifact_type=artifact_type,
            current_active_version_id=current_active_version_id,
            target_version_id=version_id,
        )
        authorization = enforce_ahi_authorization_result(
            authorize_scoped_ahi_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=self._tenant_scope_resolver,
                request=request,
            ),
            operation="apply_profile",
        )

        apply_result = executor.apply(
            package,
            tenant_id=tenant_id,
            task_class=task_class,
            version_id=version_id,
            expected_active_version_id=current_active_version_id,
        )
        return GovernedApplyProfileResult(
            apply_result=apply_result,
            authorization_evidence=authorization.evidence,
        )

    def rollback_profile(
        self,
        *,
        executor: AdaptationExecutor,
        pointer_store: ProfileActivePointerStore,
        principal: RequestIdentity,
        mutation_id: str,
        tenant_id: str,
        task_class: str,
        artifact_type: ProfileArtifactType,
    ) -> GovernedRollbackProfileResult:
        """Rollback active profile via fresh control-plane authorization."""
        validate_ahi_principal_tenant_authority(
            principal=principal,
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            operation="rollback_profile",
        )

        pointer = pointer_store.get_pointer(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
        )
        if pointer is None or pointer.previous_version_id is None:
            raise ValueError("No rollback pointer available for active profile version")

        current_active_version_id = pointer.active_version_id
        rollback_target_version_id = pointer.previous_version_id

        boundary = self._require_mutation_authorization_boundary()
        request = build_rollback_profile_mutation_request(
            principal=principal,
            tenant_id=tenant_id,
            task_class=task_class,
            mutation_id=mutation_id,
            artifact_type=artifact_type,
            current_active_version_id=current_active_version_id,
            target_previous_version_id=rollback_target_version_id,
        )
        authorization = enforce_ahi_authorization_result(
            authorize_scoped_ahi_control_plane_mutation(
                boundary=boundary,
                tenant_resolver=self._tenant_scope_resolver,
                request=request,
            ),
            operation="rollback_profile",
        )

        rollback_result = executor.rollback(
            tenant_id=tenant_id,
            task_class=task_class,
            artifact_type=artifact_type,
            expected_active_version_id=current_active_version_id,
        )
        return GovernedRollbackProfileResult(
            rollback_result=rollback_result,
            authorization_evidence=authorization.evidence,
        )

    def _require_mutation_authorization_boundary(
        self,
    ) -> ControlPlaneMutationAuthorizationBoundary:
        boundary = self._mutation_authorization_boundary
        if boundary is None:
            raise AhiGovernanceBlockedError(
                "AHI_BLOCKED_BY_MISSING_MUTATION_AUTHORIZATION_BOUNDARY",
                "control-plane mutations require ControlPlaneMutationAuthorizationBoundary",
                policy_action="DENY",
            )
        return boundary
