# © Artur Czarnecki. All rights reserved.

"""MP-3G — execution lineage integration and evidence direction tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

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
    WorkArtifactVersionRepository,
)
from intergrax.contracts.collaborative_work import (
    ArtifactContentRef,
    AuthorityDelegation,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkAuthorizationDenied,
    CreateWorkArtifactFromExecutionRequest,
    CreateWorkArtifactRequest,
    DelegationStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyLayerApplicability,
    PublishWorkArtifactVersionFromExecutionRequest,
    PublishWorkArtifactVersionRequest,
    WorkArtifactVersion,
    WorkArtifactVersionRef,
    WorkspaceMembership,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.evidence_artifact_reference import EvidenceArtifactVersionLink
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_provenance import ExecutionProvenanceRef
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_DELEGATOR = "principal-delegator"
_AUTHORITY_SCOPE = "collaborative_work.manage"
_WORK_ITEM_ID = "work-item-1"
_ARTIFACT_ID = "artifact-1"
_VERSION_1 = "artifact-version-1"
_VERSION_2 = "artifact-version-2"
_VERSION_3 = "artifact-version-3"
_VERSION_4 = "artifact-version-4"
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
    version_repo: WorkArtifactVersionRepository


def _content_ref(**overrides: object) -> ArtifactContentRef:
    payload = {
        "content_ref": "content://tenant-a/workspace-a/body-1",
        "media_type": "application/json",
        "integrity_digest": _DIGEST,
    }
    payload.update(overrides)
    return ArtifactContentRef.model_validate(payload)


def _execution(**overrides: object) -> ExecutionProvenanceRef:
    payload = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
    }
    payload.update(overrides)
    return ExecutionProvenanceRef(**payload)


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
    grant_id: str = "authority-grant-acting",
) -> object:
    from intergrax.collaborative_work.repository import CreatePrincipalAuthorityGrantCommand

    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id=grant_id,
            principal_id=principal_id,
            authority_scopes=(_AUTHORITY_SCOPE,),
        )
    )


def _seed_work_item(repo: InMemoryWorkItemRepository) -> object:
    return repo.create(
        CreateWorkItemCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
            created_by_principal_id=_ACTING,
            created_at=_NOW,
            updated_at=_NOW,
        )
    )


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
        version_repo=artifact_bundle.version,
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


def _create_from_execution_request(**overrides: object) -> CreateWorkArtifactFromExecutionRequest:
    execution = overrides.pop("execution", _execution())
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "work_artifact_id": _ARTIFACT_ID,
        "work_artifact_version_id": _VERSION_1,
        "acting_principal_id": _ACTING,
        "content_ref": _content_ref(),
        "execution": execution,
        "idempotency_key": "create-artifact-exec-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateWorkArtifactFromExecutionRequest(**payload)


def _publish_from_execution_request(**overrides: object) -> PublishWorkArtifactVersionFromExecutionRequest:
    execution = overrides.pop("execution", _execution())
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "work_artifact_id": _ARTIFACT_ID,
        "work_artifact_version_id": _VERSION_2,
        "expected_revision": INITIAL_RECORD_REVISION,
        "acting_principal_id": _ACTING,
        "content_ref": _content_ref(content_ref="content://tenant-a/workspace-a/body-2"),
        "execution": execution,
        "idempotency_key": "publish-artifact-exec-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return PublishWorkArtifactVersionFromExecutionRequest(**payload)


def _version_ref(**overrides: object) -> WorkArtifactVersionRef:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "work_artifact_id": _ARTIFACT_ID,
        "work_artifact_version_id": _VERSION_1,
    }
    payload.update(overrides)
    return WorkArtifactVersionRef(**payload)


def test_work_artifact_version_ref_strict_frozen() -> None:
    ref = _version_ref()
    assert ref.model_config["frozen"] is True
    assert ref.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        WorkArtifactVersionRef(**{**ref.model_dump(), "content_ref": "spoof"})


def test_create_from_execution_request_strict() -> None:
    request = _create_from_execution_request()
    assert request.model_config["frozen"] is True
    assert request.model_config["extra"] == "forbid"


def test_publish_from_execution_request_strict() -> None:
    request = _publish_from_execution_request()
    assert request.model_config["frozen"] is True
    assert request.model_config["extra"] == "forbid"


def test_execution_request_rejects_dict_provenance() -> None:
    provenance = _execution()
    create_payload = _create_from_execution_request().model_dump(mode="python")
    create_payload["execution"] = {
        "task_id": provenance.task_id,
        "run_id": provenance.run_id,
        "attempt_id": provenance.attempt_id,
        "execution_id": provenance.execution_id,
    }
    with pytest.raises(TypeError, match="ExecutionProvenanceRef"):
        CreateWorkArtifactFromExecutionRequest(**create_payload)
    publish_payload = _publish_from_execution_request().model_dump(mode="python")
    publish_payload["execution"] = {
        "task_id": provenance.task_id,
        "run_id": provenance.run_id,
        "attempt_id": provenance.attempt_id,
        "execution_id": provenance.execution_id,
    }
    with pytest.raises(TypeError, match="ExecutionProvenanceRef"):
        PublishWorkArtifactVersionFromExecutionRequest(**publish_payload)


def test_public_mp3c_requests_reject_execution_field() -> None:
    with pytest.raises(ValidationError):
        CreateWorkArtifactRequest(**{**_create_request().model_dump(), "execution": "spoof"})
    with pytest.raises(ValidationError):
        PublishWorkArtifactVersionRequest(**{**_publish_request().model_dump(), "execution": "spoof"})


def test_human_create_persists_execution_none() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_artifact(_create_request())
    assert created.version.execution is None


def test_human_publish_persists_execution_none() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    published = fixture.service.publish_version(_publish_request())
    assert published.version.execution is None


def test_execution_create_exact_provenance() -> None:
    fixture = _service_fixture()
    execution = _execution()
    created = fixture.service.create_artifact_from_execution(
        _create_from_execution_request(execution=execution, idempotency_key="exec-create-1"),
    )
    assert created.version.execution == execution
    assert created.version.execution is not None
    assert created.version.execution.task_id == execution.task_id
    assert created.version.execution.run_id == execution.run_id
    assert created.version.execution.attempt_id == execution.attempt_id
    assert created.version.execution.execution_id == execution.execution_id


def test_execution_publish_exact_provenance() -> None:
    fixture = _service_fixture()
    fixture.service.create_artifact(_create_request())
    execution = _execution()
    published = fixture.service.publish_version_from_execution(
        _publish_from_execution_request(execution=execution, idempotency_key="exec-publish-1"),
    )
    assert published.version.execution == execution


def test_principal_provenance_independent_from_execution() -> None:
    fixture = _service_fixture()
    execution = _execution()
    created = fixture.service.create_artifact_from_execution(
        _create_from_execution_request(
            acting_principal_id=_ACTING,
            execution=execution,
            idempotency_key="exec-principal-1",
        ),
    )
    assert created.version.created_by_principal_id == _ACTING
    assert created.version.published_by_principal_id == _ACTING
    assert created.version.execution == execution
    assert str(created.version.execution.execution_id) != _ACTING


def test_mixed_history_preserves_none_e1_none_e2() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    execution_one = _execution()
    execution_two = _execution()
    v1 = fixture.service.create_artifact(_create_request(idempotency_key="mixed-v1"))
    v2 = fixture.service.publish_version_from_execution(
        _publish_from_execution_request(
            work_artifact_version_id=_VERSION_2,
            expected_revision=INITIAL_RECORD_REVISION,
            execution=execution_one,
            idempotency_key="mixed-v2",
        ),
    )
    v3 = fixture.service.publish_version(
        _publish_request(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            idempotency_key="mixed-v3",
        ),
    )
    v4 = fixture.service.publish_version_from_execution(
        _publish_from_execution_request(
            work_artifact_version_id=_VERSION_4,
            expected_revision=2,
            execution=execution_two,
            idempotency_key="mixed-v4",
        ),
    )
    listed = fixture.version_repo.list_for_artifact(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_artifact_id=_ARTIFACT_ID,
    )
    assert listed == (v1.version, v2.version, v3.version, v4.version)
    assert [version.execution for version in listed] == [None, execution_one, None, execution_two]


def test_execution_does_not_bypass_authority_gate() -> None:
    fixture = _service_fixture(seed_authority=False)
    execution = _execution()
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_artifact_from_execution(
            _create_from_execution_request(execution=execution, idempotency_key="exec-deny"),
        )
    assert (
        fixture.version_repo.list_for_artifact(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_artifact_id=_ARTIFACT_ID,
        )
        == ()
    )


def test_execution_delegated_authority_works() -> None:
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
    execution = _execution()
    created = fixture.service.create_artifact_from_execution(
        _create_from_execution_request(
            execution=execution,
            idempotency_key="exec-delegated",
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=_acting_membership_locator(),
            delegator_principal_id=_DELEGATOR,
            delegation=_delegation_locator(resource_scope=resource_scope),
        )
    )
    assert created.version.execution == execution


def test_execution_publish_idempotency_replay() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    execution = _execution()
    fixture.service.create_artifact(_create_request())
    request = _publish_from_execution_request(
        execution=execution,
        idempotency_key="exec-idem",
    )
    first = fixture.service.publish_version_from_execution(request)
    second = fixture.service.publish_version_from_execution(request)
    assert second == first


def test_execution_publish_idempotency_replay_after_later_publication() -> None:
    fixture = _service_fixture(clock=_AdvancingClock(_NOW))
    execution = _execution()
    fixture.service.create_artifact(_create_request())
    original_request = _publish_from_execution_request(
        execution=execution,
        idempotency_key="exec-idem",
    )
    original = fixture.service.publish_version_from_execution(original_request)
    fixture.service.publish_version(
        _publish_request(
            work_artifact_version_id=_VERSION_3,
            expected_revision=1,
            idempotency_key="exec-later",
        )
    )
    replay = fixture.service.publish_version_from_execution(original_request)
    assert replay == original
    assert replay.artifact.revision == 1


def test_execution_publish_changed_provenance_same_key_conflicts() -> None:
    fixture = _service_fixture()
    execution_one = _execution()
    execution_two = _execution()
    fixture.service.create_artifact(_create_request())
    fixture.service.publish_version_from_execution(
        _publish_from_execution_request(
            execution=execution_one,
            idempotency_key="exec-idem-conflict",
        )
    )
    with pytest.raises(ArtifactPublicationIdempotencyConflict):
        fixture.service.publish_version_from_execution(
            _publish_from_execution_request(
                work_artifact_version_id=_VERSION_3,
                execution=execution_two,
                idempotency_key="exec-idem-conflict",
            )
        )


def test_evidence_association_references_version_without_mutating_it() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_artifact(_create_request())
    version = created.version
    version_ref = WorkArtifactVersionRef(
        tenant_id=version.tenant_id,
        workspace_id=version.workspace_id,
        work_item_id=version.work_item_id,
        work_artifact_id=version.work_artifact_id,
        work_artifact_version_id=version.work_artifact_version_id,
    )
    link = EvidenceArtifactVersionLink(
        evidence_id="evidence-1",
        artifact_version=version_ref,
    )
    assert link.artifact_version.work_artifact_version_id == _VERSION_1
    assert "evidence_id" not in WorkArtifactVersion.model_fields
    assert "receipt_id" not in WorkArtifactVersion.model_fields
    assert "evidence_refs" not in WorkArtifactVersion.model_fields
    assert version.execution is None


def test_evidence_artifact_version_link_strict_frozen() -> None:
    link = EvidenceArtifactVersionLink(
        evidence_id="evidence-1",
        artifact_version=_version_ref(),
    )
    assert link.model_config["frozen"] is True
    assert link.model_config["extra"] == "forbid"
    with pytest.raises(ValidationError):
        EvidenceArtifactVersionLink(**{**link.model_dump(), "metadata": {"spoof": True}})
