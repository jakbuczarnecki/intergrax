# © Artur Czarnecki. All rights reserved.

"""MP-4R4 — CollaborativeDecisionBinding service and repository conformance."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from intergrax.collaborative_work.decision_binding_service import (
    TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE,
    CollaborativeDecisionBindingService,
)
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
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
from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateArtifactWithInitialVersionCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkItemCommand,
    CreateWorkspaceMembershipCommand,
    PublishWorkArtifactVersionCommand,
    WorkItemNotFound,
    INITIAL_RECORD_REVISION,
)
from intergrax.contracts.collaborative_decision_binding import (
    CollaborativeDecisionBindingIdempotencyConflict,
    CollaborativeDecisionBindingReferenceMismatch,
    CollaborativeDecisionBindingScopeMismatch,
    CreateCollaborativeDecisionBindingRequest,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkAuthorizationDenied,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    WorkArtifactVersionRef,
    WorkItemState,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
)
from intergrax.contracts.decision_record import DecisionProposalRef, decision_lineage_ref
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
pytestmark = pytest.mark.unit


class _UnusedRuntimeEvaluator:
    def evaluate(self, *_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime policy evaluator must not be invoked in this test")

_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-a"
_WORKSPACE_B = "workspace-b"
_WORK_ITEM_ID = "work-item-1"
_ACTING = "principal-acting"
_AUTHORITY_SCOPE = work_item_resource_scope(work_item_id=_WORK_ITEM_ID)
_NOW = datetime(2026, 9, 16, 10, 0, tzinfo=UTC)
_DIGEST = "sha256:" + "a" * 64


def _identity(
    *,
    tenant_id: str = _TENANT,
    version: DecisionVersion | None = None,
) -> DecisionIdentity:
    resolved_version = version or initial_decision_version()
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=resolved_version,
        scope=DecisionScope(namespace="incident", subject="incident-1"),
        tenant_id=tenant_id,
        execution=DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
    )


def _proposal(identity: DecisionIdentity) -> DecisionProposalRef:
    return DecisionProposalRef(
        identity=identity,
        lineage_ref=decision_lineage_ref(identity.version),
    )


class _ServiceBundle:
    def __init__(
        self,
        *,
        service: CollaborativeDecisionBindingService,
        work_item_repo: object,
        version_repo: object,
        binding_repo: object,
    ) -> None:
        self.service = service
        self.work_item_repo = work_item_repo
        self.version_repo = version_repo
        self.binding_repo = binding_repo


def _profile_command(*, operation_id: str) -> CreateCollaborativeOperationPolicyProfileCommand:
    return CreateCollaborativeOperationPolicyProfileCommand(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        operation_id=operation_id,
        authority_scope=_AUTHORITY_SCOPE,
        workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
    )


def _build_in_memory_service(*, seed_authority: bool = True) -> _ServiceBundle:
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
    if seed_authority:
        authority_repo.create(
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id="grant-1",
                principal_id=_ACTING,
                authority_scopes=(_AUTHORITY_SCOPE,),
            ),
        )
    profile_repo.create(
        _profile_command(operation_id=TRUSTED_OPERATION_COLLABORATIVE_DECISION_BINDING_CREATE),
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
    service = CollaborativeDecisionBindingService(
        work_item_repository=work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=binding_repo,
        enforcement_gate=gate,
        clock=lambda: _NOW,
    )
    return _ServiceBundle(
        service=service,
        work_item_repo=work_item_repo,
        version_repo=artifact_bundle.version,
        binding_repo=binding_repo,
    )


def _seed_work_item(repo: object, *, work_item_id: str = _WORK_ITEM_ID) -> None:
    repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=work_item_id,
            created_by_principal_id=_ACTING,
            state=WorkItemState.OPEN,
            created_at=_NOW,
            updated_at=_NOW,
        ),
    )


def _binding_request(**overrides: object) -> CreateCollaborativeDecisionBindingRequest:
    identity = _identity()
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "decision_proposal": _proposal(identity),
        "acting_principal_id": _ACTING,
        "idempotency_key": "binding-idem-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateCollaborativeDecisionBindingRequest(**payload)


def test_work_item_only_binding_round_trip() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    created = bundle.service.create_binding(_binding_request())
    loaded = bundle.service.get_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        binding_id=created.binding_id,
    )
    assert loaded == created
    assert loaded is not None
    assert loaded.work_artifact_version is None
    assert loaded.decision_proposal == created.decision_proposal


def test_artifact_version_binding() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    artifact_bundle = open_in_memory_artifact_repositories()
    publication = artifact_bundle.publication.create_artifact_with_initial_version(
        CreateArtifactWithInitialVersionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            work_artifact_id="artifact-1",
            work_artifact_version_id="artifact-version-1",
            created_by_principal_id=_ACTING,
            published_by_principal_id=_ACTING,
            content_ref=ArtifactContentRef(
                content_ref="content://tenant-a/workspace-a/body",
                media_type="application/json",
                integrity_digest=_DIGEST,
            ),
            artifact_created_at=_NOW,
            artifact_updated_at=_NOW,
            version_created_at=_NOW,
            version_published_at=_NOW,
            execution=None,
        ),
    )
    bundle.service = CollaborativeDecisionBindingService(
        work_item_repository=bundle.work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=bundle.binding_repo,
        enforcement_gate=bundle.service._enforcement_gate,
        clock=lambda: _NOW,
    )
    ref = WorkArtifactVersionRef(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
        work_artifact_id="artifact-1",
        work_artifact_version_id=publication.version.work_artifact_version_id,
    )
    created = bundle.service.create_binding(
        _binding_request(work_artifact_version=ref, idempotency_key="artifact-binding"),
    )
    assert created.work_artifact_version == ref


def test_cross_work_item_artifact_rejected() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    ref = WorkArtifactVersionRef(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id="other-work-item",
        work_artifact_id="artifact-1",
        work_artifact_version_id="artifact-version-1",
    )
    with pytest.raises(ValueError, match="work_item_id"):
        bundle.service.create_binding(_binding_request(work_artifact_version=ref))


def test_cross_tenant_decision_rejected() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    identity = _identity(tenant_id=_TENANT_B)
    with pytest.raises(CollaborativeDecisionBindingScopeMismatch):
        bundle.service.create_binding(_binding_request(decision_proposal=_proposal(identity)))


def test_exact_decision_version_preserved_after_new_version_exists() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    identity_v1 = _identity(version=initial_decision_version())
    proposal_v1 = _proposal(identity_v1)
    created = bundle.service.create_binding(
        _binding_request(decision_proposal=proposal_v1, idempotency_key="v1-binding"),
    )
    identity_v2 = DecisionIdentity(
        decision_id=identity_v1.decision_id,
        version=next_decision_version(identity_v1.version),
        scope=identity_v1.scope,
        tenant_id=identity_v1.tenant_id,
        execution=identity_v1.execution,
    )
    _proposal(identity_v2)
    loaded = bundle.service.get_binding(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        binding_id=created.binding_id,
    )
    assert loaded is not None
    assert loaded.decision_proposal.identity.version.value == 1


def test_artifact_binding_stays_on_v1_after_v2_published() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    artifact_bundle = open_in_memory_artifact_repositories()
    publication = artifact_bundle.publication.create_artifact_with_initial_version(
        CreateArtifactWithInitialVersionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            work_artifact_id="artifact-2",
            work_artifact_version_id="artifact-version-v1",
            created_by_principal_id=_ACTING,
            published_by_principal_id=_ACTING,
            content_ref=ArtifactContentRef(
                content_ref="content://tenant-a/workspace-a/v1",
                media_type="application/json",
                integrity_digest=_DIGEST,
            ),
            artifact_created_at=_NOW,
            artifact_updated_at=_NOW,
            version_created_at=_NOW,
            version_published_at=_NOW,
            execution=None,
        ),
    )
    later = _NOW + timedelta(seconds=30)
    artifact_bundle.publication.publish_version(
        PublishWorkArtifactVersionCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            work_artifact_id="artifact-2",
            work_artifact_version_id="artifact-version-v2",
            expected_revision=INITIAL_RECORD_REVISION,
            created_by_principal_id=_ACTING,
            published_by_principal_id=_ACTING,
            content_ref=ArtifactContentRef(
                content_ref="content://tenant-a/workspace-a/v2",
                media_type="application/json",
                integrity_digest=_DIGEST,
            ),
            created_at=later,
            published_at=later,
            artifact_updated_at=later,
            execution=None,
        ),
    )
    ref_v1 = WorkArtifactVersionRef(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
        work_artifact_id="artifact-2",
        work_artifact_version_id=publication.version.work_artifact_version_id,
    )
    service = CollaborativeDecisionBindingService(
        work_item_repository=bundle.work_item_repo,
        work_artifact_version_repository=artifact_bundle.version,
        binding_repository=bundle.binding_repo,
        enforcement_gate=bundle.service._enforcement_gate,
        clock=lambda: _NOW,
    )
    created = service.create_binding(
        _binding_request(work_artifact_version=ref_v1, idempotency_key="artifact-v1-binding"),
    )
    assert created.work_artifact_version == ref_v1


def test_idempotent_replay_preserves_binding_id() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    identity = _identity()
    request = _binding_request(decision_proposal=_proposal(identity))
    first = bundle.service.create_binding(request)
    second = bundle.service.create_binding(request)
    assert second.binding_id == first.binding_id
    assert second == first


def test_idempotency_conflict_on_semantic_change() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    bundle.service.create_binding(_binding_request())
    other_identity = _identity()
    with pytest.raises(CollaborativeDecisionBindingIdempotencyConflict):
        bundle.service.create_binding(
            _binding_request(decision_proposal=_proposal(other_identity)),
        )


def test_semantic_duplicate_returns_same_binding() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    identity = _identity()
    proposal = _proposal(identity)
    first = bundle.service.create_binding(
        _binding_request(decision_proposal=proposal, idempotency_key="first-key"),
    )
    second = bundle.service.create_binding(
        _binding_request(decision_proposal=proposal, idempotency_key="second-key"),
    )
    assert second.binding_id == first.binding_id


def test_tenant_isolation_on_get() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    created = bundle.service.create_binding(_binding_request())
    assert (
        bundle.service.get_binding(
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE,
            binding_id=created.binding_id,
        )
        is None
    )


def test_workspace_isolation_on_list() -> None:
    bundle = _build_in_memory_service()
    _seed_work_item(bundle.work_item_repo)
    bundle.service.create_binding(_binding_request())
    assert (
        bundle.service.list_bindings_for_work_item(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE_B,
            work_item_id=_WORK_ITEM_ID,
        )
        == ()
    )


def test_unauthorized_create_denied() -> None:
    bundle = _build_in_memory_service(seed_authority=False)
    _seed_work_item(bundle.work_item_repo)
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        bundle.service.create_binding(_binding_request())


def test_missing_work_item_fail_closed() -> None:
    bundle = _build_in_memory_service()
    with pytest.raises(WorkItemNotFound):
        bundle.service.create_binding(_binding_request())


def test_sqlite_binding_round_trip(tmp_path: Path) -> None:
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "bindings.sqlite"))
    try:
        bundle.work_item.create(
            CreateWorkItemCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                work_item_id=_WORK_ITEM_ID,
                created_by_principal_id=_ACTING,
                state=WorkItemState.OPEN,
                created_at=_NOW,
                updated_at=_NOW,
            ),
        )
        identity = _identity()
        command_repo = bundle.decision_binding
        from intergrax.collaborative_work.repository import CreateCollaborativeDecisionBindingCommand
        from intergrax.contracts.collaborative_decision_binding import mint_collaborative_decision_binding_id

        created = command_repo.create(
            CreateCollaborativeDecisionBindingCommand(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                binding_id=mint_collaborative_decision_binding_id(idempotency_key="sqlite-1"),
                work_item_id=_WORK_ITEM_ID,
                work_artifact_version=None,
                decision_proposal=_proposal(identity),
                created_by_principal_id=_ACTING,
                created_at=_NOW,
                idempotency_key="sqlite-1",
            ),
        )
        loaded = command_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            binding_id=created.binding_id,
        )
        assert loaded == created
    finally:
        bundle.close()


def test_service_does_not_import_sqlite_backend() -> None:
    import ast
    from pathlib import Path

    path = Path("intergrax/collaborative_work/decision_binding_service.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]
    assert not any(name and "sqlite" in name for name in imports)
