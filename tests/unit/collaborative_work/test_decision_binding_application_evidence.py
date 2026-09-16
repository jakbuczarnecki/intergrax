# © Artur Czarnecki. All rights reserved.

"""MP-4R5 — application-level decision binding create evidence adoption."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.decision_binding_application import (
    CollaborativeDecisionBindingApplicationService,
    CollaborativeDecisionBindingEvidenceAdoption,
)
from intergrax.collaborative_work.decision_binding_service import (
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
    CollaborativeDecisionBindingService,
)
from intergrax.collaborative_work.functional_evidence_projection import (
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
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.decision_binding_composition import (
    build_collaborative_decision_binding_application_service,
    build_collaborative_decision_binding_service,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    WorkItemNotFound,
)
from intergrax.contracts.collaborative_decision_binding import CreateCollaborativeDecisionBindingRequest
from intergrax.contracts.collaborative_functional_evidence_projection import (
    COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID,
    CollaborativeDecisionBindingCreateOutcomeProjection,
    CollaborativeFunctionalEvidenceProjectionStrategy,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfileStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
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
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.functional_evidence.models import (
    PipelineEvidenceKind,
    PipelineOperationStatus,
    PlatformFunctionalEvidence,
)
from intergrax.contracts.functional_evidence.persistence import (
    FunctionalEvidencePersistence,
    FunctionalEvidencePersistenceConflictError,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_SEED = "mp4r5-app-evidence"
_TASK: TaskId = canonical_task_id_for_tests(_SEED)
_RUN: RunId = canonical_run_id_for_tests(_SEED)
_ATTEMPT: AttemptId = mint_attempt_id()
_EXECUTION: ExecutionId = mint_execution_id()
_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_WORK_ITEM_ID = "work-item-1"
_ACTING = "principal-acting"
_NOW = datetime(2026, 9, 16, 10, 0, tzinfo=UTC)
_AUTHORITY_SCOPE = work_item_resource_scope(work_item_id=_WORK_ITEM_ID)


class _UnusedRuntimeEvaluator:
    def evaluate(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime policy evaluator must not be invoked in this test")


class _RecordingPersistence(FunctionalEvidencePersistence):
    def __init__(self) -> None:
        self.items: list[PlatformFunctionalEvidence] = []
        self._by_id: dict[str, PlatformFunctionalEvidence] = {}

    def append(self, evidence: PlatformFunctionalEvidence) -> PlatformFunctionalEvidence:
        key = str(evidence.evidence_id)
        existing = self._by_id.get(key)
        if existing is not None:
            if existing.model_dump() != evidence.model_dump():
                raise FunctionalEvidencePersistenceConflictError("conflict")
            return existing
        self._by_id[key] = evidence
        self.items.append(evidence)
        return evidence

    def query_evidence(self, request):  # type: ignore[no-untyped-def]
        raise NotImplementedError


class _SpyProjectionStrategy(CollaborativeFunctionalEvidenceProjectionStrategy):
    def __init__(self, delegate: CollaborativeFunctionalEvidenceProjectionStrategy) -> None:
        self.delegate = delegate
        self.projections: list[CollaborativeDecisionBindingCreateOutcomeProjection] = []

    def project_decision_binding_create_outcome(
        self,
        projection: CollaborativeDecisionBindingCreateOutcomeProjection,
    ) -> PlatformFunctionalEvidence:
        self.projections.append(projection)
        return self.delegate.project_decision_binding_create_outcome(projection)

    def project_decision_binding_association(self, binding, *, execution_correlation=None):  # type: ignore[no-untyped-def]
        return self.delegate.project_decision_binding_association(
            binding,
            execution_correlation=execution_correlation,
        )


def _identity() -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=initial_decision_version(),
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id=_TENANT,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _proposal() -> DecisionProposalRef:
    identity = _identity()
    return DecisionProposalRef(identity=identity, lineage_ref=decision_lineage_ref(identity.version))


def _profile_command() -> CreateCollaborativeOperationPolicyProfileCommand:
    return CreateCollaborativeOperationPolicyProfileCommand(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
        authority_scope=_AUTHORITY_SCOPE,
        workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
    )


def _build_gate_and_service() -> tuple[
    CollaborativeWorkEnforcementGate,
    CollaborativeDecisionBindingService,
    InMemoryWorkItemRepository,
]:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    work_item_repo = InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()
    binding_repo = InMemoryCollaborativeDecisionBindingRepository()
    membership_repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        ),
    )
    authority_repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="grant-1",
            principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
        ),
    )
    profile_repo.create(_profile_command())
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
    service = build_collaborative_decision_binding_service(
        work_item_repository=work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=binding_repo,
        enforcement_gate=gate,
        clock=lambda: _NOW,
    )
    return gate, service, work_item_repo


def _application(
    *,
    persistence: _RecordingPersistence,
    strategy: CollaborativeFunctionalEvidenceProjectionStrategy,
    gate: CollaborativeWorkEnforcementGate,
    service: CollaborativeDecisionBindingService,
) -> CollaborativeDecisionBindingApplicationService:
    return build_collaborative_decision_binding_application_service(
        binding_service=service,
        evidence_adoption=CollaborativeDecisionBindingEvidenceAdoption(
            persistence=persistence,
            strategy=strategy,
        ),
        clock=lambda: _NOW,
    )


def _correlation() -> object:
    provenance = ExecutionProvenanceRef(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    return execution_correlation_from_provenance(tenant_id=_TENANT, execution=provenance)


def _binding_request(**overrides: object) -> CreateCollaborativeDecisionBindingRequest:
    identity = _identity()
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "decision_proposal": DecisionProposalRef(
            identity=identity,
            lineage_ref=decision_lineage_ref(identity.version),
        ),
        "acting_principal_id": _ACTING,
        "idempotency_key": "idem-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateCollaborativeDecisionBindingRequest(**payload)


def test_composition_builders_wire_application_boundary() -> None:
    gate, service, _ = _build_gate_and_service()
    app = build_collaborative_decision_binding_application_service(
        binding_service=service,
        evidence_adoption=None,
        clock=lambda: _NOW,
    )
    assert isinstance(app, CollaborativeDecisionBindingApplicationService)


def test_success_emits_single_operation_outcome() -> None:
    from intergrax.collaborative_work.functional_evidence_projection import (
        DefaultCollaborativeFunctionalEvidenceProjection,
    )

    gate, service, work_item_repo = _build_gate_and_service()
    from intergrax.collaborative_work.repository import CreateWorkItemCommand
    from intergrax.contracts.collaborative_work import WorkItemState

    work_item_repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            state=WorkItemState.OPEN,
            created_by_principal_id=_ACTING,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    persistence = _RecordingPersistence()
    app = _application(
        persistence=persistence,
        strategy=DefaultCollaborativeFunctionalEvidenceProjection(),
        gate=gate,
        service=service,
    )
    correlation = _correlation()
    app.create_binding(_binding_request(), execution_correlation=correlation)
    assert len(persistence.items) == 1
    evidence = persistence.items[0]
    assert evidence.kind is PipelineEvidenceKind.OPERATION_OUTCOME
    assert evidence.operation_outcome is not None
    assert evidence.operation_outcome.status is PipelineOperationStatus.SUCCEEDED
    assert evidence.operation_outcome.operation_name == COLLABORATIVE_DECISION_BINDING_CREATE_OPERATION_ID
    assert evidence.scope.task_id == _TASK
    assert evidence.scope.run_id == _RUN
    assert evidence.scope.attempt_id == _ATTEMPT
    assert evidence.scope.execution_id == _EXECUTION


def test_failed_before_binding_without_synthetic_binding() -> None:
    from intergrax.collaborative_work.functional_evidence_projection import (
        DefaultCollaborativeFunctionalEvidenceProjection,
    )

    gate, service, _work_item_repo = _build_gate_and_service()
    persistence = _RecordingPersistence()
    spy = _SpyProjectionStrategy(DefaultCollaborativeFunctionalEvidenceProjection())
    app = _application(persistence=persistence, strategy=spy, gate=gate, service=service)
    with pytest.raises(WorkItemNotFound):
        app.create_binding(_binding_request(), execution_correlation=_correlation())
    assert len(persistence.items) == 1
    assert persistence.items[0].operation_outcome is not None
    assert persistence.items[0].operation_outcome.status is PipelineOperationStatus.FAILED
    assert len(spy.projections) == 1
    assert spy.projections[0].binding is None


def test_no_correlation_skips_evidence() -> None:
    from intergrax.collaborative_work.functional_evidence_projection import (
        DefaultCollaborativeFunctionalEvidenceProjection,
    )

    gate, service, work_item_repo = _build_gate_and_service()
    from intergrax.collaborative_work.repository import CreateWorkItemCommand
    from intergrax.contracts.collaborative_work import WorkItemState

    work_item_repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            state=WorkItemState.OPEN,
            created_by_principal_id=_ACTING,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    persistence = _RecordingPersistence()
    app = _application(
        persistence=persistence,
        strategy=DefaultCollaborativeFunctionalEvidenceProjection(),
        gate=gate,
        service=service,
    )
    app.create_binding(_binding_request(), execution_correlation=None)
    assert persistence.items == []


def test_custom_persistence_and_strategy_injected() -> None:
    from intergrax.collaborative_work.functional_evidence_projection import (
        DefaultCollaborativeFunctionalEvidenceProjection,
    )

    gate, service, work_item_repo = _build_gate_and_service()
    from intergrax.collaborative_work.repository import CreateWorkItemCommand
    from intergrax.contracts.collaborative_work import WorkItemState

    work_item_repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            state=WorkItemState.OPEN,
            created_by_principal_id=_ACTING,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )
    persistence = _RecordingPersistence()
    spy = _SpyProjectionStrategy(DefaultCollaborativeFunctionalEvidenceProjection())
    app = _application(persistence=persistence, strategy=spy, gate=gate, service=service)
    app.create_binding(_binding_request(), execution_correlation=_correlation(), evidence_id=mint_event_id())
    assert len(spy.projections) == 1
