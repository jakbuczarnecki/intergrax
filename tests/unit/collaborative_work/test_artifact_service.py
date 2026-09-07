# © Artur Czarnecki. All rights reserved.

"""MP-3C — authoritative WorkArtifact publication service tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intergrax.collaborative_work.artifact_service import (
    CollaborativeWorkArtifactService,
    TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
    TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
)
from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkItemRepository,
    InMemoryWorkspaceMembershipRepository,
    open_in_memory_artifact_repositories,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    ArtifactPublicationIdempotencyConflict,
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateWorkItemCommand,
    INITIAL_RECORD_REVISION,
    WorkArtifactIdempotencyConflict,
    WorkArtifactNotFound,
    WorkArtifactRevisionConflict,
    WorkArtifactTemporalConflict,
    WorkItemNotFound,
    WorkArtifactVersionRepository,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    AuthorityDelegation,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkAuthorizationDenied,
    CreateWorkArtifactRequest,
    CreateWorkItemRequest,
    DelegationStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    PublishWorkArtifactVersionRequest,
    WorkspaceMembership,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision
from intergrax.collaborative_work.service import CollaborativeWorkService

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE = "workspace-a"
_WORKSPACE_B = "workspace-b"
_ACTING = "principal-acting"
_DELEGATOR = "principal-delegator"
_AUTHORITY_SCOPE = "collaborative_work.manage"
_WORK_ITEM_ID = "work-item-1"
_ARTIFACT_ID = "artifact-1"
_VERSION_1 = "artifact-version-1"
_VERSION_2 = "artifact-version-2"
_VERSION_3 = "artifact-version-3"
_DIGEST = "sha256:" + ("a" * 64)
_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal artifact mutations",
            policy_rule_id="test.unexpected_runtime",
        )


class _AdvancingClock:
    def __init__(self, start: datetime, *, step: timedelta = timedelta(seconds=1)) -> None:
        self._current = start
        self._step = step

    def __call__(self) -> datetime:
        value = self._current
        self._current += self._step
        return value


@dataclass(frozen=True, slots=True)
class _ArtifactServiceFixture:
    service: CollaborativeWorkArtifactService
    membership_repo: InMemoryWorkspaceMembershipRepository
    authority_repo: InMemoryPrincipalAuthorityRepository
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository
    delegation_repo: InMemoryAuthorityDelegationRepository
    work_item_repo: InMemoryWorkItemRepository
    version_repo: WorkArtifactVersionRepository
    publication_repo: object


def _content_ref(**overrides: object) -> ArtifactContentRef:
    payload = {
        "content_ref": "content://tenant-a/workspace-a/body-1",
        "media_type": "application/json",
        "integrity_digest": _DIGEST,
    }
    payload.update(overrides)
    return ArtifactContentRef.model_validate(payload)


def _profile_command(*, operation_id: str, **overrides: object) -> CreateCollaborativeOperationPolicyProfileCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation_id": operation_id,
        "authority_scope": _AUTHORITY_SCOPE,
        "workspace_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "resource_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "runtime_policy_applicability": PolicyLayerApplicability.NOT_APPLICABLE,
        "resource_requirement": OperationPolicyRequirement.NOT_APPLICABLE,
        "meaningful_side_effect_requirement": OperationPolicyRequirement.NOT_APPLICABLE,
        "status": CollaborativeOperationPolicyProfileStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativeOperationPolicyProfileCommand(**payload)


def _seed_artifact_profiles(profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository) -> None:
    for operation_id in (
        TRUSTED_OPERATION_WORK_ARTIFACT_CREATE,
        TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH,
    ):
        profile_repo.create(_profile_command(operation_id=operation_id))


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    principal_id: str = _ACTING,
    membership_id: str = "membership-acting",
) -> WorkspaceMembership:
    from intergrax.collaborative_work.repository import CreateWorkspaceMembershipCommand

    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id=membership_id,
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )


def _seed_authority(
    repo: InMemoryPrincipalAuthorityRepository,
    *,
    principal_id: str = _ACTING,
    authority_scopes: tuple[str, ...] = (_AUTHORITY_SCOPE,),
    grant_id: str = "authority-grant-acting",
) -> object:
    from intergrax.collaborative_work.repository import CreatePrincipalAuthorityGrantCommand

    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id=grant_id,
            principal_id=principal_id,
            authority_scopes=authority_scopes,
        )
    )


def _seed_work_item(repo: InMemoryWorkItemRepository, **overrides: object) -> object:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "created_by_principal_id": _ACTING,
        "created_at": _NOW,
        "updated_at": _NOW,
    }
    payload.update(overrides)
    return repo.create(CreateWorkItemCommand(**payload))


def _delegation_locator(**overrides: object) -> AuthorityDelegation:
    resource_scope = work_item_resource_scope(work_item_id=_WORK_ITEM_ID)
    payload = {
        "delegation_id": "delegation-1",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "delegator_principal_id": _DELEGATOR,
        "delegate_principal_id": _ACTING,
        "authority_scopes": (_AUTHORITY_SCOPE,),
        "resource_scope": resource_scope,
        "status": DelegationStatus.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return AuthorityDelegation.model_validate(payload)


def _acting_membership_locator(**overrides: object) -> WorkspaceMembership:
    payload = {
        "membership_id": "membership-acting",
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "principal_id": _ACTING,
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
        "revision": 0,
    }
    payload.update(overrides)
    return WorkspaceMembership.model_validate(payload)


def _service_fixture(
    *,
    seed_profiles: bool = True,
    seed_membership: bool = True,
    seed_authority: bool = True,
    seed_work_item: bool = True,
    clock: _AdvancingClock | None = None,
) -> _ArtifactServiceFixture:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    work_item_repo = InMemoryWorkItemRepository()
    artifact_bundle = open_in_memory_artifact_repositories()

    if seed_membership:
        _seed_membership(membership_repo)
    if seed_authority:
        _seed_authority(authority_repo)
    if seed_profiles:
        _seed_artifact_profiles(profile_repo)
    if seed_work_item:
        _seed_work_item(work_item_repo)

    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    service = CollaborativeWorkArtifactService(
        work_item_repository=work_item_repo,
        work_artifact_repository=artifact_bundle.artifact,
        artifact_publication_repository=artifact_bundle.publication,
        enforcement_gate=gate,
        clock=clock or _AdvancingClock(_NOW),
    )
    return _ArtifactServiceFixture(
        service=service,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        profile_repo=profile_repo,
        delegation_repo=delegation_repo,
        work_item_repo=work_item_repo,
        version_repo=artifact_bundle.version,
        publication_repo=artifact_bundle.publication,
    )


def _create_request(**overrides: object) -> CreateWorkArtifactRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "work_artifact_id": _ARTIFACT_ID,
        "work_artifact_version_id": _VERSION_1,
        "acting_principal_id": _ACTING,
        "content_ref": _content_ref(),
        "idempotency_key": "create-artifact-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateWorkArtifactRequest(**payload)


def _publish_request(**overrides: object) -> PublishWorkArtifactVersionRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "work_artifact_id": _ARTIFACT_ID,
        "work_artifact_version_id": _VERSION_2,
        "expected_revision": INITIAL_RECORD_REVISION,
        "acting_principal_id": _ACTING,
        "content_ref": _content_ref(content_ref="content://tenant-a/workspace-a/body-2"),
        "idempotency_key": "publish-artifact-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return PublishWorkArtifactVersionRequest(**payload)


# --- request contract strictness ---


def test_create_request_contract_strictness() -> None:
    request = _create_request()
    assert request.model_config["frozen"] is True
    assert request.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        CreateWorkArtifactRequest(**{**_create_request().model_dump(), "created_by_principal_id": "spoof"})
    with pytest.raises(ValidationError):
        CreateWorkArtifactRequest(**{**_create_request().model_dump(), "execution": "spoof"})


def test_publish_request_contract_strictness() -> None:
    request = _publish_request()
    assert request.model_config["frozen"] is True
    assert request.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        PublishWorkArtifactVersionRequest(**{**_publish_request().model_dump(), "expected_revision": -1})
    with pytest.raises(ValidationError):
        PublishWorkArtifactVersionRequest(**{**_publish_request().model_dump(), "published_by_principal_id": "spoof"})


def test_create_request_rejects_canonical_principal_with_embedded_membership() -> None:
    with pytest.raises(ValidationError):
        _create_request(
            membership_resolution_mode=MembershipResolutionMode.CANONICAL_PRINCIPAL,
            membership=_acting_membership_locator(),
        )


# --- create ---


def test_create_authorized_success() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_artifact(_create_request())
    assert created.artifact.revision == INITIAL_RECORD_REVISION
    assert created.version.work_artifact_version_id == _VERSION_1
    assert created.version.created_by_principal_id == _ACTING
    assert created.version.published_by_principal_id == _ACTING


def test_create_requires_parent_work_item() -> None:
    fixture = _service_fixture(seed_work_item=False)
    with pytest.raises(WorkItemNotFound):
        fixture.service.create_artifact(_create_request())


def test_create_wrong_tenant_or_workspace_not_found() -> None:
    fixture = _service_fixture()
    with pytest.raises(WorkItemNotFound):
        fixture.service.create_artifact(_create_request(tenant_id=_TENANT_B))
    with pytest.raises(WorkItemNotFound):
        fixture.service.create_artifact(_create_request(workspace_id=_WORKSPACE_B))


def test_create_missing_profile_denied_without_record() -> None:
    fixture = _service_fixture(seed_profiles=False)
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.create_artifact(_create_request())
    assert exc.value.enforcement_result.operation_id == TRUSTED_OPERATION_WORK_ARTIFACT_CREATE
    assert (
        fixture.version_repo.list_for_artifact(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_artifact_id=_ARTIFACT_ID,
        )
        == ()
    )


def test_create_missing_membership_denied() -> None:
    fixture = _service_fixture(seed_membership=False)
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_artifact(_create_request())


def test_create_insufficient_authority_denied() -> None:
    fixture = _service_fixture(seed_authority=False)
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_artifact(_create_request())


def test_create_delegated_correct_work_item_allowed() -> None:
    fixture = _service_fixture(seed_membership=False, seed_authority=False)
    _seed_membership(fixture.membership_repo, principal_id=_ACTING)
    _seed_membership(fixture.membership_repo, principal_id=_DELEGATOR, membership_id="membership-delegator")
    resource_scope = work_item_resource_scope(work_item_id=_WORK_ITEM_ID)
    fixture.delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
            resource_scope=resource_scope,
        )
    )
    _seed_authority(fixture.authority_repo, principal_id=_DELEGATOR, grant_id="authority-grant-delegator")
    created = fixture.service.create_artifact(
        _create_request(
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=_acting_membership_locator(),
            delegator_principal_id=_DELEGATOR,
            delegation=_delegation_locator(resource_scope=resource_scope),
        )
    )
    assert created.artifact.work_item_id == _WORK_ITEM_ID


def test_create_delegation_wrong_work_item_denied() -> None:
    fixture = _service_fixture(seed_membership=False, seed_authority=False)
    _seed_membership(fixture.membership_repo, principal_id=_ACTING)
    _seed_membership(fixture.membership_repo, principal_id=_DELEGATOR, membership_id="membership-delegator")
    wrong_scope = work_item_resource_scope(work_item_id="other-work-item")
    fixture.delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
            resource_scope=wrong_scope,
        )
    )
    _seed_authority(fixture.authority_repo, principal_id=_DELEGATOR, grant_id="authority-grant-delegator")
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_artifact(
            _create_request(
                membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                membership=_acting_membership_locator(),
                delegator_principal_id=_DELEGATOR,
                delegation=_delegation_locator(resource_scope=wrong_scope),
            )
        )


def test_create_advancing_clock_idempotency_replay() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    request = _create_request(idempotency_key="create-idem")
    first = fixture.service.create_artifact(request)
    second = fixture.service.create_artifact(request)
    assert second == first
    listed = fixture.version_repo.list_for_artifact(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_artifact_id=_ARTIFACT_ID,
    )
    assert listed == (first.version,)


def test_create_changed_semantic_intent_same_key_conflicts() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request(idempotency_key="create-idem"))
    with pytest.raises(WorkArtifactIdempotencyConflict):
        fixture.service.create_artifact(
            _create_request(
                work_artifact_id="artifact-other",
                idempotency_key="create-idem",
            )
        )


def test_create_naive_clock_rejected() -> None:
    fixture = _service_fixture()
    with patch.object(fixture.service, "_clock", return_value=datetime(2026, 9, 7, 12, 0)):
        with pytest.raises(ValueError, match="timezone-aware"):
            fixture.service.create_artifact(_create_request())


# --- publish ---


def test_publish_authorized_success() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    published = fixture.service.publish_version(_publish_request())
    assert published.artifact.revision == INITIAL_RECORD_REVISION + 1
    assert published.version.work_artifact_version_id == _VERSION_2
    assert published.version.created_by_principal_id == _ACTING
    assert published.version.published_by_principal_id == _ACTING
    assert published.version.execution is None


def test_publish_stored_artifact_scope_used_for_wrong_work_item() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    with pytest.raises(WorkArtifactNotFound):
        fixture.service.publish_version(_publish_request(work_item_id="other-work-item"))


def test_publish_stale_revision_propagated() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    fixture.service.publish_version(_publish_request(expected_revision=INITIAL_RECORD_REVISION))
    with pytest.raises(WorkArtifactRevisionConflict):
        fixture.service.publish_version(
            _publish_request(
                work_artifact_version_id=_VERSION_3,
                expected_revision=INITIAL_RECORD_REVISION,
                idempotency_key="publish-stale",
            )
        )


def test_publish_missing_profile_denied_without_mutation() -> None:
    fixture = _service_fixture(seed_profiles=False)
    fixture.profile_repo.create(_profile_command(operation_id=TRUSTED_OPERATION_WORK_ARTIFACT_CREATE))
    fixture.service.create_artifact(_create_request())
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.publish_version(_publish_request())
    assert exc.value.enforcement_result.operation_id == TRUSTED_OPERATION_WORK_ARTIFACT_PUBLISH


def test_publish_wrong_tenant_or_workspace_not_found() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    with pytest.raises(WorkArtifactNotFound):
        fixture.service.publish_version(_publish_request(tenant_id=_TENANT_B))
    with pytest.raises(WorkArtifactNotFound):
        fixture.service.publish_version(_publish_request(workspace_id=_WORKSPACE_B))


def test_publish_advancing_clock_idempotency_replay() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    fixture.service.create_artifact(_create_request())
    request = _publish_request(idempotency_key="publish-idem")
    first = fixture.service.publish_version(request)
    second = fixture.service.publish_version(request)
    assert second == first


def test_publish_retry_after_later_publication_returns_original_result() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    fixture.service.create_artifact(_create_request())
    original_request = _publish_request(idempotency_key="publish-idem")
    original = fixture.service.publish_version(original_request)
    fixture.service.publish_version(
        _publish_request(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            idempotency_key="publish-later",
        )
    )
    replay = fixture.service.publish_version(original_request)
    assert replay == original
    assert replay.artifact.revision == 1


def test_publish_changed_semantic_intent_same_key_conflicts() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    fixture.service.publish_version(_publish_request(idempotency_key="publish-idem"))
    with pytest.raises(ArtifactPublicationIdempotencyConflict):
        fixture.service.publish_version(
            _publish_request(
                work_artifact_version_id=_VERSION_3,
                idempotency_key="publish-idem",
            )
        )


def test_publish_preserves_previous_versions() -> None:
    fixture = _service_fixture()
    initial = fixture.service.create_artifact(_create_request())
    published = fixture.service.publish_version(_publish_request())
    listed = fixture.version_repo.list_for_artifact(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_artifact_id=_ARTIFACT_ID,
    )
    assert listed == (initial.version, published.version)


def test_publish_temporal_conflict_propagated() -> None:
    fixture = _service_fixture()
    initial = fixture.service.create_artifact(_create_request())
    with patch.object(
        fixture.publication_repo,
        "publish_version",
        side_effect=WorkArtifactTemporalConflict("temporal conflict"),
    ):
        with pytest.raises(WorkArtifactTemporalConflict):
            fixture.service.publish_version(_publish_request(expected_revision=initial.artifact.revision))


def test_authority_revoked_after_success_blocks_retry() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    request = _create_request(idempotency_key="create-idem")
    fixture.service.create_artifact(request)
    from intergrax.collaborative_work.repository import (
        PrincipalAuthorityGrantScopeKey,
        UpdatePrincipalAuthorityGrantCommand,
    )
    from intergrax.contracts.collaborative_work import AuthorityGrantStatus

    grant = fixture.authority_repo.get_for_principal(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        principal_id=_ACTING,
    )
    assert grant is not None
    fixture.authority_repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id=grant.authority_grant_id,
            ),
            expected_revision=grant.revision,
            authority_scopes=(_AUTHORITY_SCOPE,),
            status=AuthorityGrantStatus.REVOKED,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_artifact(request)


def test_shared_work_service_constructor_unchanged() -> None:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    work_item_repo = InMemoryWorkItemRepository()
    from intergrax.collaborative_work.in_memory_repository import InMemoryAssignmentRepository

    gate = CollaborativeWorkEnforcementGate(
        profile_repository=profile_repo,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership_repo,
            delegation_repository=delegation_repo,
            principal_authority_repository=authority_repo,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy_repo),
        runtime_policy_evaluator=_UnusedRuntimeEvaluator(),
    )
    shared_service = CollaborativeWorkService(
        work_item_repository=work_item_repo,
        assignment_repository=InMemoryAssignmentRepository(),
        enforcement_gate=gate,
    )
    assert shared_service is not None


def test_publication_port_delegation_only() -> None:
    fixture = _service_fixture()
    with (
        patch.object(
            fixture.publication_repo,
            "create_artifact_with_initial_version",
            wraps=fixture.publication_repo.create_artifact_with_initial_version,
        ) as create_spy,
        patch.object(
            fixture.publication_repo,
            "publish_version",
            wraps=fixture.publication_repo.publish_version,
        ) as publish_spy,
    ):
        fixture.service.create_artifact(_create_request())
        fixture.service.publish_version(_publish_request())
    assert create_spy.call_count == 1
    assert publish_spy.call_count == 1
