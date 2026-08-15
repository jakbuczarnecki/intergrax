# © Artur Czarnecki. All rights reserved.

"""Shared repository contract suite for in-memory and durable adapters."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.persistence import open_sqlite_collaborative_work_repositories
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    PrincipalAuthorityGrantAlreadyExists,
    UpdateAuthorityDelegationCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipAlreadyExists,
    WorkspaceMembershipRevisionConflict,
    WorkspaceMembershipScopeKey,
    AuthorityDelegationScopeKey,
    CollaborativePolicyRuleAlreadyExists,
    UpdateCollaborativeOperationPolicyProfileCommand,
    CollaborativeOperationPolicyProfileScopeKey,
    CollaborativeOperationPolicyProfileRevisionConflict,
)
from intergrax.contracts.collaborative_work import (
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativePolicyRuleStatus,
    DelegationStatus,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembershipRole,
)
from intergrax.contracts.runtime_policy import PolicyAction

pytestmark = pytest.mark.unit

_TENANT_A = "tenant-a"
_TENANT_B = "tenant-b"
_WORKSPACE_A = "workspace-a"
_WORKSPACE_B = "workspace-b"
_VALID_FROM = datetime(2026, 1, 1, tzinfo=UTC)
_VALID_UNTIL = datetime(2026, 12, 31, tzinfo=UTC)


@pytest.fixture(params=("memory", "sqlite"))
def membership_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryWorkspaceMembershipRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "membership.sqlite"))
    try:
        yield bundle.membership
    finally:
        bundle.close()


@pytest.fixture(params=("memory", "sqlite"))
def delegation_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryAuthorityDelegationRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "delegation.sqlite"))
    try:
        yield bundle.delegation
    finally:
        bundle.close()


@pytest.fixture(params=("memory", "sqlite"))
def authority_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryPrincipalAuthorityRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "authority.sqlite"))
    try:
        yield bundle.principal_authority
    finally:
        bundle.close()


@pytest.fixture(params=("memory", "sqlite"))
def policy_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryCollaborativePolicyRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "policy.sqlite"))
    try:
        yield bundle.policy
    finally:
        bundle.close()


@pytest.fixture(params=("memory", "sqlite"))
def profile_repo(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        yield InMemoryCollaborativeOperationPolicyProfileRepository()
        return
    bundle = open_sqlite_collaborative_work_repositories(str(tmp_path / "profile.sqlite"))
    try:
        yield bundle.operation_profile
    finally:
        bundle.close()


def _membership_command(**overrides: object) -> CreateWorkspaceMembershipCommand:
    payload = {
        "tenant_id": _TENANT_A,
        "workspace_id": _WORKSPACE_A,
        "membership_id": "membership-1",
        "principal_id": "principal-1",
        "role": WorkspaceMembershipRole.MEMBER,
        "status": MembershipStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateWorkspaceMembershipCommand(**payload)


def test_membership_create_get_revision_and_isolation(membership_repo: object) -> None:
    created = membership_repo.create(_membership_command())  # type: ignore[attr-defined]
    assert created.revision == INITIAL_RECORD_REVISION
    loaded = membership_repo.get(  # type: ignore[attr-defined]
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        membership_id="membership-1",
    )
    assert loaded == created
    assert (
        membership_repo.get(  # type: ignore[attr-defined]
            tenant_id=_TENANT_B,
            workspace_id=_WORKSPACE_B,
            membership_id="membership-1",
        )
        is None
    )


def test_membership_duplicate_and_stale_revision(membership_repo: object) -> None:
    created = membership_repo.create(_membership_command())  # type: ignore[attr-defined]
    with pytest.raises(WorkspaceMembershipAlreadyExists):
        membership_repo.create(_membership_command())  # type: ignore[attr-defined]
    with pytest.raises(WorkspaceMembershipRevisionConflict):
        membership_repo.update(  # type: ignore[attr-defined]
            UpdateWorkspaceMembershipCommand(
                scope=WorkspaceMembershipScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    membership_id="membership-1",
                ),
                expected_revision=created.revision + 1,
                role=WorkspaceMembershipRole.ADMIN,
                status=MembershipStatus.SUSPENDED,
            )
        )


def test_membership_idempotency_replay_after_update(membership_repo: object) -> None:
    command = _membership_command(idempotency_key="membership-idem")
    created = membership_repo.create(command)  # type: ignore[attr-defined]
    membership_repo.update(  # type: ignore[attr-defined]
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                membership_id="membership-1",
            ),
            expected_revision=created.revision,
            role=WorkspaceMembershipRole.ADMIN,
            status=MembershipStatus.SUSPENDED,
        )
    )
    replayed = membership_repo.create(command)  # type: ignore[attr-defined]
    assert replayed == created
    assert replayed.role is WorkspaceMembershipRole.MEMBER


def test_delegation_create_update_idempotency(delegation_repo: object) -> None:
    command = CreateAuthorityDelegationCommand(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        delegation_id="delegation-1",
        delegator_principal_id="delegator",
        delegate_principal_id="delegate",
        authority_scopes=("workspace.read",),
        status=DelegationStatus.ACTIVE,
        idempotency_key="delegation-idem",
    )
    created = delegation_repo.create(command)  # type: ignore[attr-defined]
    assert delegation_repo.create(command) == created  # type: ignore[attr-defined]
    updated = delegation_repo.update(  # type: ignore[attr-defined]
        UpdateAuthorityDelegationCommand(
            scope=AuthorityDelegationScopeKey(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                delegation_id="delegation-1",
            ),
            expected_revision=created.revision,
            authority_scopes=("workspace.write",),
            resource_scope="resource-1",
            valid_from=_VALID_FROM,
            valid_until=_VALID_UNTIL,
            status=DelegationStatus.REVOKED,
        )
    )
    assert updated.revision == created.revision + 1
    replayed = delegation_repo.create(command)  # type: ignore[attr-defined]
    assert replayed == created


def test_authority_grant_principal_uniqueness(authority_repo: object) -> None:
    command = CreatePrincipalAuthorityGrantCommand(
        tenant_id=_TENANT_A,
        workspace_id=_WORKSPACE_A,
        authority_grant_id="grant-1",
        principal_id="principal-1",
        authority_scopes=("workspace.read",),
        status=AuthorityGrantStatus.ACTIVE,
    )
    authority_repo.create(command)  # type: ignore[attr-defined]
    with pytest.raises(PrincipalAuthorityGrantAlreadyExists):
        authority_repo.create(  # type: ignore[attr-defined]
            CreatePrincipalAuthorityGrantCommand(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                authority_grant_id="grant-2",
                principal_id="principal-1",
                authority_scopes=("workspace.write",),
                status=AuthorityGrantStatus.ACTIVE,
            )
        )


def test_policy_exact_key_uniqueness(policy_repo: object) -> None:
    policy_repo.create(  # type: ignore[attr-defined]
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            policy_rule_id="rule-1",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope="document.delete",
            action=PolicyAction.ALLOW,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    with pytest.raises(CollaborativePolicyRuleAlreadyExists):
        policy_repo.create(  # type: ignore[attr-defined]
            CreateCollaborativePolicyRuleCommand(
                tenant_id=_TENANT_A,
                workspace_id=_WORKSPACE_A,
                policy_rule_id="rule-2",
                layer=PolicyCompositionLayer.WORKSPACE_POLICY,
                authority_scope="document.delete",
                action=PolicyAction.DENY,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )


def test_profile_revision_increment(profile_repo: object) -> None:
    created = profile_repo.create(  # type: ignore[attr-defined]
        CreateCollaborativeOperationPolicyProfileCommand(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            operation_id="operation-1",
            authority_scope="document.delete",
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            status=CollaborativeOperationPolicyProfileStatus.ACTIVE,
        )
    )
    with pytest.raises(CollaborativeOperationPolicyProfileRevisionConflict):
        profile_repo.update(  # type: ignore[attr-defined]
            UpdateCollaborativeOperationPolicyProfileCommand(
                scope=CollaborativeOperationPolicyProfileScopeKey(
                    tenant_id=_TENANT_A,
                    workspace_id=_WORKSPACE_A,
                    operation_id="operation-1",
                ),
                expected_revision=created.revision + 1,
                authority_scope="document.delete",
                workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
                resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
                resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
                status=CollaborativeOperationPolicyProfileStatus.DISABLED,
            )
        )


def test_sqlite_persistence_across_reinstantiation(tmp_path: Path) -> None:
    db_path = str(tmp_path / "persist.sqlite")
    first = open_sqlite_collaborative_work_repositories(db_path)
    first.membership.create(_membership_command())
    first.close()

    second = open_sqlite_collaborative_work_repositories(db_path)
    try:
        loaded = second.membership.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            membership_id="membership-1",
        )
        assert loaded is not None
        assert loaded.revision == INITIAL_RECORD_REVISION
        caps = second.membership.capabilities
        assert caps.durable is True
        assert caps.reference_only is False
        assert caps.backend_id == "collaborative_work.sqlite"
    finally:
        second.close()


def test_sqlite_concurrent_update_one_wins(tmp_path: Path) -> None:
    db_path = str(tmp_path / "concurrent.sqlite")
    setup = open_sqlite_collaborative_work_repositories(db_path)
    created = setup.membership.create(_membership_command())
    setup.close()

    errors: list[BaseException] = []
    barrier = threading.Barrier(2)

    def attempt(expected_revision: int) -> None:
        repo_bundle = open_sqlite_collaborative_work_repositories(db_path)
        try:
            barrier.wait(timeout=5)
            repo_bundle.membership.update(
                UpdateWorkspaceMembershipCommand(
                    scope=WorkspaceMembershipScopeKey(
                        tenant_id=_TENANT_A,
                        workspace_id=_WORKSPACE_A,
                        membership_id="membership-1",
                    ),
                    expected_revision=expected_revision,
                    role=WorkspaceMembershipRole.ADMIN,
                    status=MembershipStatus.SUSPENDED,
                )
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            repo_bundle.close()

    threads = [
        threading.Thread(target=attempt, args=(created.revision,)),
        threading.Thread(target=attempt, args=(created.revision,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 1
    assert isinstance(errors[0], WorkspaceMembershipRevisionConflict)

    verify = open_sqlite_collaborative_work_repositories(db_path)
    try:
        final = verify.membership.get(
            tenant_id=_TENANT_A,
            workspace_id=_WORKSPACE_A,
            membership_id="membership-1",
        )
        assert final is not None
        assert final.revision == created.revision + 1
    finally:
        verify.close()
