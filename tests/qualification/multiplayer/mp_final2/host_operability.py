# © Artur Czarnecki. All rights reserved.

"""Composition harness for MP-FINAL-2 Multiplayer → Evidence → Diagnostics proofs.

Wires real Collaborative Decision Binding application + canonical Evidence Plane
persistence with platform Functional Diagnostics interpretation. Composition owns
implementation selection; source Multiplayer modules do not import Diagnostics.
"""

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
from intergrax.collaborative_work.decision_binding_service import (
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.functional_evidence_projection import (
    DefaultCollaborativeFunctionalEvidenceProjection,
    execution_correlation_from_provenance,
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
    CreateCollaborativeDecisionBindingCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
    WorkItemRepository,
)
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBinding,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_functional_evidence_projection import (
    COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfileStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkItemState,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    initial_decision_version,
    mint_decision_id,
)
from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.functional_evidence.correlation import (
    FunctionalEvidenceExecutionCorrelation,
)
from intergrax.contracts.functional_evidence.models import PipelineOperationStatus
from intergrax.contracts.functional_evidence.persistence import FunctionalEvidencePersistence
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyDecision
from intergrax.runtime.diagnostics.diagnostic_read_service import DiagnosticReadService
from intergrax.runtime.diagnostics.functional_diagnostic_analyzer import (
    FunctionalDiagnosticAnalyzer,
)
from intergrax.runtime.diagnostics.functional_diagnostic_identity import (
    FunctionalDiagnosticCheckId,
    FunctionalDiagnosticSpecificationId,
)
from intergrax.runtime.diagnostics.functional_diagnostic_specification import (
    FunctionalDiagnosticCheck,
    FunctionalDiagnosticRequirement,
    FunctionalDiagnosticRequirementKind,
    FunctionalDiagnosticSpecification,
    OperationOutcomeStatusRequirement,
    validate_functional_diagnostic_specification,
)
from intergrax.runtime.diagnostics.functional_operator_projection import (
    FunctionalDiagnosticOperatorProjection,
    FunctionalOperatorProjector,
)
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.observability.functional_evidence.in_memory_functional_evidence_persistence import (
    InMemoryFunctionalEvidencePersistence,
)
from intergrax.runtime.observability.memory_causal_evidence_persistence import (
    InMemoryCausalEvidencePersistence,
)
from intergrax.runtime.observability.reconstruction import ExecutionReconstructor
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests
from testing_support.runtime.diagnostics.problem_persistence_test_support import (
    read_service_for_tests,
)

_NOW = datetime(2026, 9, 20, 8, 0, tzinfo=UTC)
_SEED = "mp-final2-operability"
_DEFAULT_TENANT = "mp-final2-tenant-a"
_DEFAULT_WORKSPACE = "mp-final2-workspace-a"
_DEFAULT_WORK_ITEM = "mp-final2-work-item-1"
_DEFAULT_ACTING = "mp-final2-principal-acting"
_DEFAULT_AUTHORITY_SCOPE = work_item_resource_scope(work_item_id=_DEFAULT_WORK_ITEM)

MP_FINAL2_SPECIFICATION_ID = FunctionalDiagnosticSpecificationId(
    "fdspec_00000000000000000000000000a1f201",
)
MP_FINAL2_CHECK_BINDING_CREATE = FunctionalDiagnosticCheckId(
    "fdcheck_00000000000000000000000000a1f201",
)


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        raise AssertionError("runtime policy evaluator must not be invoked in MP-FINAL-2 harness")


class FailingCreateBindingRepository(InMemoryCollaborativeDecisionBindingRepository):
    """Legal repository adapter seam: simulate persistence failure after ALLOW."""

    def create(
        self,
        command: CreateCollaborativeDecisionBindingCommand,
    ) -> CollaborativeDecisionBinding:
        raise RuntimeError("simulated collaborative decision binding persistence failure")


@dataclass(frozen=True, slots=True)
class OperabilityHost:
    """Composition-selected Multiplayer + Evidence + Diagnostics read surfaces."""

    tenant_id: str
    workspace_id: str
    work_item_id: str
    acting_principal_id: str
    application: CollaborativeDecisionBindingApplicationService
    evidence_persistence: FunctionalEvidencePersistence
    projection_strategy: CollaborativeFunctionalEvidenceProjectionStrategy
    analyzer: FunctionalDiagnosticAnalyzer
    operator_projector: FunctionalOperatorProjector
    diagnostic_read_service: DiagnosticReadService
    problem_persistence: InMemoryProblemPersistence
    task_id: TaskId
    run_id: RunId
    attempt_id: AttemptId
    execution_id: ExecutionId
    clock: Callable[[], datetime]


def build_decision_binding_create_operability_specification() -> FunctionalDiagnosticSpecification:
    """Diagnostics-owned interpretation profile for Multiplayer binding-create evidence."""
    return validate_functional_diagnostic_specification(
        FunctionalDiagnosticSpecification(
            specification_id=MP_FINAL2_SPECIFICATION_ID,
            version=1,
            checks=(
                FunctionalDiagnosticCheck(
                    check_id=MP_FINAL2_CHECK_BINDING_CREATE,
                    requirement=FunctionalDiagnosticRequirement(
                        kind=FunctionalDiagnosticRequirementKind.OPERATION_OUTCOME_STATUS,
                        operation_outcome_status=OperationOutcomeStatusRequirement(
                            operation_id=COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
                            expected_status=PipelineOperationStatus.SUCCEEDED,
                        ),
                    ),
                    pass_claim="Collaborative decision binding create succeeded.",
                    fail_claim="Collaborative decision binding create failed.",
                    insufficient_claim="No collaborative decision binding create outcome evidence.",
                ),
            ),
        ),
    )


def build_operability_host(
    *,
    tenant_id: str = _DEFAULT_TENANT,
    workspace_id: str = _DEFAULT_WORKSPACE,
    work_item_id: str = _DEFAULT_WORK_ITEM,
    acting_principal_id: str = _DEFAULT_ACTING,
    seed_work_item: bool = True,
    seed_authority: bool = True,
    binding_repository: CollaborativeDecisionBindingRepository | None = None,
    work_item_repository: WorkItemRepository | None = None,
    evidence_persistence: FunctionalEvidencePersistence | None = None,
    projection_strategy: CollaborativeFunctionalEvidenceProjectionStrategy | None = None,
    task_id: TaskId | None = None,
    run_id: RunId | None = None,
    attempt_id: AttemptId | None = None,
    execution_id: ExecutionId | None = None,
) -> OperabilityHost:
    """Compose Multiplayer binding app + Evidence Plane + Functional Diagnostics."""
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    work_item_repo = work_item_repository or InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()
    binding_repo = binding_repository or InMemoryCollaborativeDecisionBindingRepository()
    authority_scope = work_item_resource_scope(work_item_id=work_item_id)

    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{acting_principal_id}",
            principal_id=acting_principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    if seed_authority:
        authority_repo.create(
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                authority_grant_id=f"grant-{acting_principal_id}",
                principal_id=acting_principal_id,
                authority_scopes=(authority_scope,),
            ),
        )
    profile_repo.create(
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
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
    if seed_work_item:
        work_item_repo.create(
            CreateWorkItemCommand(
                tenant_id=tenant_id,
                workspace_id=workspace_id,
                work_item_id=work_item_id,
                state=WorkItemState.OPEN,
                created_by_principal_id=acting_principal_id,
                created_at=_NOW,
                updated_at=_NOW,
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
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    binding_service = build_collaborative_decision_binding_service(
        work_item_repository=work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=binding_repo,
        enforcement_gate=gate,
        clock=lambda: _NOW,
    )
    resolved_persistence = evidence_persistence or InMemoryFunctionalEvidencePersistence(
        cursor_secret=b"mp-final2-functional-evidence-cursor-v1",
    )
    resolved_strategy = projection_strategy or DefaultCollaborativeFunctionalEvidenceProjection()
    application = build_collaborative_decision_binding_application_service(
        binding_service=binding_service,
        evidence_adoption=CollaborativeDecisionBindingEvidenceAdoption(
            persistence=resolved_persistence,
            strategy=resolved_strategy,
        ),
        clock=lambda: _NOW,
    )

    problem_persistence = InMemoryProblemPersistence()
    diagnostic_read = read_service_for_tests(
        problem_persistence,
        ExecutionReconstructor(
            runtime_events=InMemoryRuntimeEventStore(),
            causal_evidence=InMemoryCausalEvidencePersistence(),
        ),
    )

    resolved_task = task_id or canonical_task_id_for_tests(_SEED)
    resolved_run = run_id or canonical_run_id_for_tests(_SEED)
    return OperabilityHost(
        tenant_id=tenant_id,
        workspace_id=workspace_id,
        work_item_id=work_item_id,
        acting_principal_id=acting_principal_id,
        application=application,
        evidence_persistence=resolved_persistence,
        projection_strategy=resolved_strategy,
        analyzer=FunctionalDiagnosticAnalyzer(resolved_persistence),
        operator_projector=FunctionalOperatorProjector(),
        diagnostic_read_service=diagnostic_read,
        problem_persistence=problem_persistence,
        task_id=resolved_task,
        run_id=resolved_run,
        attempt_id=attempt_id or mint_attempt_id(),
        execution_id=execution_id or mint_execution_id(),
        clock=lambda: _NOW,
    )


def execution_correlation_for(host: OperabilityHost) -> FunctionalEvidenceExecutionCorrelation:
    return execution_correlation_from_provenance(
        tenant_id=host.tenant_id,
        execution=ExecutionProvenanceRef(
            task_id=host.task_id,
            run_id=host.run_id,
            attempt_id=host.attempt_id,
            execution_id=host.execution_id,
        ),
    )


def binding_create_request(
    host: OperabilityHost,
    *,
    idempotency_key: str = "mp-final2-idem-1",
) -> CreateCollaborativeDecisionBindingRequest:
    identity = DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="mp-final2", subject="incident-1"),
        tenant_id=host.tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )
    return CreateCollaborativeDecisionBindingRequest(
        tenant_id=host.tenant_id,
        workspace_id=host.workspace_id,
        work_item_id=host.work_item_id,
        decision_proposal=DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        acting_principal_id=host.acting_principal_id,
        idempotency_key=idempotency_key,
        membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
    )


def interpret_binding_create_operability(
    host: OperabilityHost,
) -> FunctionalDiagnosticOperatorProjection:
    """Public Diagnostics interpretation → operator-facing typed projection."""
    analysis = host.analyzer.analyze(
        tenant_id=host.tenant_id,
        task_id=host.task_id,
        run_id=host.run_id,
        attempt_id=host.attempt_id,
        specification=build_decision_binding_create_operability_specification(),
    )
    return host.operator_projector.project(analysis)


__all__ = [
    "FailingCreateBindingRepository",
    "MP_FINAL2_CHECK_BINDING_CREATE",
    "MP_FINAL2_SPECIFICATION_ID",
    "OperabilityHost",
    "binding_create_request",
    "build_decision_binding_create_operability_specification",
    "build_operability_host",
    "execution_correlation_for",
    "interpret_binding_create_operability",
]
