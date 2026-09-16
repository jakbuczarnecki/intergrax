# © Artur Czarnecki. All rights reserved.

"""MP-4R7 qualification composition root — contracts and replaceable providers only."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.decision_binding_application import (
    CollaborativeDecisionBindingApplicationService,
    CollaborativeDecisionBindingEvidenceAdoption,
)
from intergrax.collaborative_work.decision_binding_composition import (
    build_collaborative_decision_binding_application_service,
    build_collaborative_decision_binding_service,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.functional_evidence_projection import (
    DefaultCollaborativeFunctionalEvidenceProjection,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeDecisionBindingRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkItemRepository,
    InMemoryWorkspaceMembershipRepository,
    open_in_memory_artifact_repositories,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CollaborativeDecisionBindingRepository,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    WorkItemRepository,
)
from intergrax.collaborative_work.decision_binding_service import (
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfileStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.decision_authorization import DecisionGovernancePolicyContext
from intergrax.contracts.execution_continuation import ExecutionContinuationPort
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId, mint_attempt_id
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence
from intergrax.runtime.execution.continuation.composition import (
    ExecutionEngineContinuationDependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    backing_execution_continuation_state_store,
    wire_execution_continuation_state_store,
)
from intergrax.runtime.observability.functional_evidence.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from testing_support.mp4r7_enterprise_integration.contracts import MP4R7_SCENARIO_SEED
from testing_support.mp4r7_enterprise_integration.decision_helpers import (
    Mp4R7RecordingHumanReviewPort,
    mp4r7_governance_policy_context,
)

_TENANT = "tenant-mp4r7"
_WORKSPACE = "workspace-mp4r7"
_ACTING = "principal-mp4r7-acting"
_WORK_ITEM = "work-item-mp4r7"
_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)


class _AllowRuntimePolicy:
    def evaluate(self, *_args: object, **_kwargs: object) -> object:
        from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

        return PolicyDecision(action=PolicyAction.ALLOW, reason="mp4r7 qualification allow")


@dataclass(frozen=True, slots=True)
class Mp4R7EnterpriseIntegrationComposition:
    tenant_id: str
    workspace_id: str
    work_item_id: str
    acting_principal_id: str
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    binding_repository: CollaborativeDecisionBindingRepository
    work_item_repository: WorkItemRepository
    binding_application: CollaborativeDecisionBindingApplicationService
    enforcement_gate: CollaborativeWorkEnforcementGate
    continuation_dependencies: ExecutionEngineContinuationDependencies
    continuation_port: ExecutionContinuationPort
    continuation_state_store: ExecutionContinuationStateStore
    continuation_backing: ExecutionContinuationDurableBacking | None
    evidence_persistence: FunctionalEvidencePersistence
    human_review_port: Mp4R7RecordingHumanReviewPort
    current_execution_policy_context: DecisionGovernancePolicyContext
    clock: Callable[[], datetime]


def open_mp4r7_enterprise_integration_composition(
    *,
    evidence_persistence: FunctionalEvidencePersistence | None = None,
    binding_repository: CollaborativeDecisionBindingRepository | None = None,
    durable_continuation: bool = False,
    current_execution_policy_context: DecisionGovernancePolicyContext | None = None,
) -> Mp4R7EnterpriseIntegrationComposition:
    """Assemble replaceable in-memory/SQLite-capable providers for enterprise integration proof."""
    task_id = canonical_task_id_for_tests(MP4R7_SCENARIO_SEED)
    run_id = canonical_run_id_for_tests(MP4R7_SCENARIO_SEED)
    attempt_id = mint_attempt_id()
    from intergrax.contracts.execution_identity import mint_execution_id

    execution_id = mint_execution_id()
    evidence = evidence_persistence or InMemoryFunctionalEvidencePersistence(cursor_secret=b"x" * 32)
    binding_repo = binding_repository or InMemoryCollaborativeDecisionBindingRepository()
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    work_item_repo = InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()
    authority_scope = work_item_resource_scope(work_item_id=_WORK_ITEM)
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-mp4r7",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-mp4r7",
            principal_id=_ACTING,
            authority_scopes=(authority_scope,),
        ),
    )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
            authority_scope=authority_scope,
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        ),
    )
    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=_AllowRuntimePolicy(),
    )
    binding_service = build_collaborative_decision_binding_service(
        work_item_repository=work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=binding_repo,
        enforcement_gate=gate,
        clock=lambda: _NOW,
    )
    projection = DefaultCollaborativeFunctionalEvidenceProjection()
    binding_application = build_collaborative_decision_binding_application_service(
        binding_service=binding_service,
        evidence_adoption=CollaborativeDecisionBindingEvidenceAdoption(
            persistence=evidence,
            strategy=projection,
        ),
        clock=lambda: _NOW,
    )
    if durable_continuation:
        continuation_backing = ExecutionContinuationDurableBacking()
        resolved_store = backing_execution_continuation_state_store(continuation_backing)
    else:
        continuation_backing = None
        resolved_store = wire_execution_continuation_state_store(state_store=None)
    continuation_deps = wire_execution_engine_continuation_dependencies(
        state_store=resolved_store,
    )
    return Mp4R7EnterpriseIntegrationComposition(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM,
        acting_principal_id=_ACTING,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        binding_repository=binding_repo,
        work_item_repository=work_item_repo,
        binding_application=binding_application,
        enforcement_gate=gate,
        continuation_dependencies=continuation_deps,
        continuation_port=continuation_deps.continuation,
        continuation_state_store=resolved_store,
        continuation_backing=continuation_backing,
        evidence_persistence=evidence,
        human_review_port=Mp4R7RecordingHumanReviewPort(),
        current_execution_policy_context=(
            current_execution_policy_context or mp4r7_governance_policy_context()
        ),
        clock=lambda: _NOW,
    )


__all__ = [
    "Mp4R7EnterpriseIntegrationComposition",
    "open_mp4r7_enterprise_integration_composition",
]
