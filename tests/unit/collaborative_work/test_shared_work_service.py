# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-2C — authoritative Shared Work service tests."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAssignmentRepository,
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkItemRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    INITIAL_RECORD_REVISION,
    UpdatePrincipalAuthorityGrantCommand,
    PrincipalAuthorityGrantScopeKey,
    WorkItemAlreadyExists,
    WorkItemNotFound,
    WorkItemRevisionConflict,
    AssignmentNotFound,
    AssignmentRevisionConflict,
)
from intergrax.collaborative_work.service import (
    CollaborativeWorkService,
    TRUSTED_OPERATION_ASSIGNMENT_CREATE,
    TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
    TRUSTED_OPERATION_WORK_ITEM_CREATE,
    TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
)
from intergrax.contracts.collaborative_work import (
    AuthorityDelegation,
    AuthorityGrantStatus,
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkAuthorizationDenied,
    CollaborativeWorkLifecycleError,
    CreateAssignmentRequest,
    CreateWorkItemRequest,
    DelegationStatus,
    MembershipResolutionMode,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    TransitionAssignmentRequest,
    TransitionWorkItemRequest,
    WorkItemState,
    AssignmentState,
    WorkspaceMembership,
    WorkspaceMembershipRole,
    work_item_resource_scope,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectRequest
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_DELEGATOR = "principal-delegator"
_TARGET = "principal-target"
_AUTHORITY_SCOPE = "collaborative_work.manage"
_WORK_ITEM_ID = "work-item-1"
_NOW = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(minutes=5)


class _UnusedRuntimeEvaluator:
    def evaluate_meaningful_side_effect(
        self,
        request: MeaningfulSideEffectRequest,
    ) -> PolicyDecision:
        _ = request
        return PolicyDecision(
            action=PolicyAction.DENY,
            reason="runtime evaluator must not run for internal shared-work mutations",
            policy_rule_id="test.unexpected_runtime",
        )


@dataclass(frozen=True, slots=True)
class _ServiceFixture:
    service: CollaborativeWorkService
    membership_repo: InMemoryWorkspaceMembershipRepository
    authority_repo: InMemoryPrincipalAuthorityRepository
    policy_repo: InMemoryCollaborativePolicyRepository
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository
    delegation_repo: InMemoryAuthorityDelegationRepository
    work_item_repo: InMemoryWorkItemRepository
    assignment_repo: InMemoryAssignmentRepository


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


def _seed_shared_work_profiles(profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository) -> None:
    for operation_id in (
        TRUSTED_OPERATION_WORK_ITEM_CREATE,
        TRUSTED_OPERATION_WORK_ITEM_TRANSITION,
        TRUSTED_OPERATION_ASSIGNMENT_CREATE,
        TRUSTED_OPERATION_ASSIGNMENT_TRANSITION,
    ):
        profile_repo.create(_profile_command(operation_id=operation_id))


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    principal_id: str = _ACTING,
    membership_id: str = "membership-acting",
) -> WorkspaceMembership:
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
    status: AuthorityGrantStatus = AuthorityGrantStatus.ACTIVE,
) -> object:
    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id=grant_id,
            principal_id=principal_id,
            authority_scopes=authority_scopes,
            status=status,
        )
    )


def _seed_delegated_memberships(repo: InMemoryWorkspaceMembershipRepository) -> None:
    _seed_membership(repo, principal_id=_ACTING, membership_id="membership-acting")
    _seed_membership(repo, principal_id=_DELEGATOR, membership_id="membership-delegator")


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
) -> _ServiceFixture:
    membership_repo = InMemoryWorkspaceMembershipRepository()
    authority_repo = InMemoryPrincipalAuthorityRepository()
    policy_repo = InMemoryCollaborativePolicyRepository()
    profile_repo = InMemoryCollaborativeOperationPolicyProfileRepository()
    delegation_repo = InMemoryAuthorityDelegationRepository()
    work_item_repo = InMemoryWorkItemRepository()
    assignment_repo = InMemoryAssignmentRepository()

    if seed_membership:
        _seed_membership(membership_repo)
    if seed_authority:
        _seed_authority(authority_repo)
    if seed_profiles:
        _seed_shared_work_profiles(profile_repo)

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
    service = CollaborativeWorkService(
        work_item_repository=work_item_repo,
        assignment_repository=assignment_repo,
        enforcement_gate=gate,
        clock=lambda: _LATER,
    )
    return _ServiceFixture(
        service=service,
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        policy_repo=policy_repo,
        profile_repo=profile_repo,
        delegation_repo=delegation_repo,
        work_item_repo=work_item_repo,
        assignment_repo=assignment_repo,
    )


def _create_work_item_request(**overrides: object) -> CreateWorkItemRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "acting_principal_id": _ACTING,
        "idempotency_key": "create-work-item-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateWorkItemRequest(**payload)


def _transition_work_item_request(**overrides: object) -> TransitionWorkItemRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "work_item_id": _WORK_ITEM_ID,
        "expected_revision": INITIAL_RECORD_REVISION,
        "target_state": WorkItemState.ACTIVE,
        "acting_principal_id": _ACTING,
        "idempotency_key": "transition-work-item-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return TransitionWorkItemRequest(**payload)


def _create_assignment_request(**overrides: object) -> CreateAssignmentRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "assignment_id": "assignment-1",
        "work_item_id": _WORK_ITEM_ID,
        "principal_id": _TARGET,
        "acting_principal_id": _ACTING,
        "idempotency_key": "create-assignment-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return CreateAssignmentRequest(**payload)


def _transition_assignment_request(**overrides: object) -> TransitionAssignmentRequest:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "assignment_id": "assignment-1",
        "work_item_id": _WORK_ITEM_ID,
        "expected_revision": INITIAL_RECORD_REVISION,
        "target_state": AssignmentState.COMPLETED,
        "acting_principal_id": _ACTING,
        "idempotency_key": "transition-assignment-1",
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return TransitionAssignmentRequest(**payload)


def _create_work_item(fixture: _ServiceFixture, **overrides: object) -> object:
    return fixture.service.create_work_item(_create_work_item_request(**overrides))


# --- create work item ---


def test_create_work_item_authorized_success() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_work_item(_create_work_item_request())
    assert created.revision == INITIAL_RECORD_REVISION
    assert created.state is WorkItemState.OPEN
    assert created.created_by_principal_id == _ACTING


def test_create_work_item_uses_trusted_operation_id_and_resource_scope() -> None:
    fixture = _service_fixture()
    created = fixture.service.create_work_item(_create_work_item_request())
    assert created.work_item_id == _WORK_ITEM_ID
    assert (
        fixture.work_item_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        == created
    )


def test_create_work_item_missing_profile_denied_without_record() -> None:
    fixture = _service_fixture(seed_profiles=False)
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.create_work_item(_create_work_item_request())
    assert exc.value.enforcement_result.operation_id == TRUSTED_OPERATION_WORK_ITEM_CREATE
    assert exc.value.enforcement_result.profile_revision is None
    assert (
        fixture.work_item_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        is None
    )


def test_create_work_item_inactive_profile_denied_without_record() -> None:
    fixture = _service_fixture(seed_profiles=False)
    fixture.profile_repo.create(
        _profile_command(
            operation_id=TRUSTED_OPERATION_WORK_ITEM_CREATE,
            status=CollaborativeOperationPolicyProfileStatus.DISABLED,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_work_item(_create_work_item_request())
    assert (
        fixture.work_item_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        is None
    )


def test_create_work_item_authority_deny_without_record() -> None:
    fixture = _service_fixture(seed_authority=False)
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.create_work_item(_create_work_item_request())
    assert exc.value.enforcement_result.composition.decision.action is PolicyAction.DENY
    assert (
        fixture.work_item_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        is None
    )


def test_create_work_item_require_human_without_record() -> None:
    fixture = _service_fixture(seed_profiles=False)
    fixture.profile_repo.create(
        _profile_command(
            operation_id=TRUSTED_OPERATION_WORK_ITEM_CREATE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
        )
    )
    fixture.policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="policy-hitl",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_AUTHORITY_SCOPE,
            action=PolicyAction.REQUIRE_HUMAN,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.create_work_item(_create_work_item_request())
    assert exc.value.enforcement_result.composition.decision.action is PolicyAction.REQUIRE_HUMAN
    assert (
        fixture.work_item_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            work_item_id=_WORK_ITEM_ID,
        )
        is None
    )


def test_create_work_item_escalate_without_record() -> None:
    fixture = _service_fixture(seed_profiles=False)
    fixture.profile_repo.create(
        _profile_command(
            operation_id=TRUSTED_OPERATION_WORK_ITEM_CREATE,
            workspace_policy_applicability=PolicyLayerApplicability.REQUIRED,
        )
    )
    fixture.policy_repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="policy-escalate",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_AUTHORITY_SCOPE,
            action=PolicyAction.ESCALATE,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied) as exc:
        fixture.service.create_work_item(_create_work_item_request())
    assert exc.value.enforcement_result.composition.decision.action is PolicyAction.ESCALATE


def test_create_work_item_idempotency_replay_still_evaluates_authority() -> None:
    fixture = _service_fixture()
    request = _create_work_item_request(idempotency_key="replay-key")
    first = fixture.service.create_work_item(request)
    second = fixture.service.create_work_item(request)
    assert second == first

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
        fixture.service.create_work_item(request)


def test_create_work_item_duplicate_without_valid_replay_raises_already_exists() -> None:
    fixture = _service_fixture()
    fixture.service.create_work_item(_create_work_item_request(idempotency_key="first-key"))
    with pytest.raises(WorkItemAlreadyExists):
        fixture.service.create_work_item(_create_work_item_request(idempotency_key="second-key"))


# --- work item transition ---


def test_transition_work_item_open_to_active() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    updated = fixture.service.transition_work_item(
        _transition_work_item_request(expected_revision=created.revision)
    )
    assert updated.state is WorkItemState.ACTIVE
    assert updated.revision == created.revision + 1


def test_transition_work_item_active_to_completed() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    active = fixture.service.transition_work_item(
        _transition_work_item_request(expected_revision=created.revision)
    )
    completed = fixture.service.transition_work_item(
        _transition_work_item_request(
            expected_revision=active.revision,
            target_state=WorkItemState.COMPLETED,
        )
    )
    assert completed.state is WorkItemState.COMPLETED


def test_transition_work_item_completed_to_active_reopen() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    active = fixture.service.transition_work_item(
        _transition_work_item_request(expected_revision=created.revision)
    )
    completed = fixture.service.transition_work_item(
        _transition_work_item_request(
            expected_revision=active.revision,
            target_state=WorkItemState.COMPLETED,
        )
    )
    reopened = fixture.service.transition_work_item(
        _transition_work_item_request(
            expected_revision=completed.revision,
            target_state=WorkItemState.ACTIVE,
        )
    )
    assert reopened.state is WorkItemState.ACTIVE


def test_transition_work_item_cancelled_to_active_reopen() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    cancelled = fixture.service.transition_work_item(
        _transition_work_item_request(
            expected_revision=created.revision,
            target_state=WorkItemState.CANCELLED,
        )
    )
    reopened = fixture.service.transition_work_item(
        _transition_work_item_request(
            expected_revision=cancelled.revision,
            target_state=WorkItemState.ACTIVE,
        )
    )
    assert reopened.state is WorkItemState.ACTIVE


def test_transition_work_item_invalid_transition_rejected() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    with pytest.raises(CollaborativeWorkLifecycleError):
        fixture.service.transition_work_item(
            _transition_work_item_request(
                expected_revision=created.revision,
                target_state=WorkItemState.COMPLETED,
            )
        )


def test_transition_work_item_stale_revision_conflict() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    fixture.service.transition_work_item(
        _transition_work_item_request(expected_revision=created.revision)
    )
    with pytest.raises(WorkItemRevisionConflict):
        fixture.service.transition_work_item(
            _transition_work_item_request(expected_revision=created.revision)
        )


def test_transition_work_item_repository_race_conflict_propagates() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    barrier = threading.Barrier(2)
    results: list[BaseException | object] = []

    def attempt() -> None:
        barrier.wait()
        try:
            results.append(
                fixture.service.transition_work_item(
                    _transition_work_item_request(expected_revision=created.revision)
                )
            )
        except BaseException as exc:
            results.append(exc)

    threads = [threading.Thread(target=attempt), threading.Thread(target=attempt)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if not isinstance(item, BaseException)]
    conflicts = [item for item in results if isinstance(item, WorkItemRevisionConflict)]
    assert len(successes) == 1
    assert len(conflicts) == 1


def test_transition_work_item_denial_leaves_state_unchanged() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    fixture.authority_repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id="authority-grant-acting",
            ),
            expected_revision=0,
            authority_scopes=("collaborative_work.other",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.transition_work_item(
            _transition_work_item_request(expected_revision=created.revision)
        )
    unchanged = fixture.work_item_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        work_item_id=_WORK_ITEM_ID,
    )
    assert unchanged == created


def test_transition_work_item_scope_mismatch_rejected() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
    with pytest.raises(WorkItemNotFound):
        fixture.service.transition_work_item(
            _transition_work_item_request(
                workspace_id="other-workspace",
                expected_revision=created.revision,
            )
        )


def test_transition_work_item_missing_raises_not_found() -> None:
    fixture = _service_fixture()
    with pytest.raises(WorkItemNotFound):
        fixture.service.transition_work_item(_transition_work_item_request())


# --- assignment create ---


def test_create_assignment_authorized_success() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    created = fixture.service.create_assignment(_create_assignment_request())
    assert created.revision == INITIAL_RECORD_REVISION
    assert created.principal_id == _TARGET
    assert created.created_by_principal_id == _ACTING


def test_create_assignment_requires_parent_work_item() -> None:
    fixture = _service_fixture()
    with pytest.raises(WorkItemNotFound):
        fixture.service.create_assignment(_create_assignment_request())


def test_create_assignment_multiple_for_same_work_item() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    first = fixture.service.create_assignment(_create_assignment_request(assignment_id="assignment-1"))
    second = fixture.service.create_assignment(
        _create_assignment_request(
            assignment_id="assignment-2",
            principal_id="principal-other",
            idempotency_key="create-assignment-2",
        )
    )
    assert first.work_item_id == second.work_item_id == _WORK_ITEM_ID
    assert first.principal_id != second.principal_id


def test_create_assignment_target_principal_may_differ_from_acting() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    created = fixture.service.create_assignment(
        _create_assignment_request(principal_id=_TARGET, acting_principal_id=_ACTING)
    )
    assert created.principal_id == _TARGET
    assert created.created_by_principal_id == _ACTING


def test_create_assignment_denied_without_record() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    fixture.authority_repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id="authority-grant-acting",
            ),
            expected_revision=0,
            authority_scopes=("collaborative_work.other",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_assignment(_create_assignment_request())
    assert (
        fixture.assignment_repo.get(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            assignment_id="assignment-1",
        )
        is None
    )


# --- assignment transition ---


def test_transition_assignment_active_to_completed() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    assignment = fixture.service.create_assignment(_create_assignment_request())
    updated = fixture.service.transition_assignment(
        _transition_assignment_request(expected_revision=assignment.revision)
    )
    assert updated.state is AssignmentState.COMPLETED


def test_transition_assignment_active_to_revoked() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    assignment = fixture.service.create_assignment(_create_assignment_request())
    updated = fixture.service.transition_assignment(
        _transition_assignment_request(
            expected_revision=assignment.revision,
            target_state=AssignmentState.REVOKED,
        )
    )
    assert updated.state is AssignmentState.REVOKED


def test_transition_assignment_invalid_terminal_transition_rejected() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    assignment = fixture.service.create_assignment(_create_assignment_request())
    completed = fixture.service.transition_assignment(
        _transition_assignment_request(expected_revision=assignment.revision)
    )
    with pytest.raises(CollaborativeWorkLifecycleError):
        fixture.service.transition_assignment(
            _transition_assignment_request(
                expected_revision=completed.revision,
                target_state=AssignmentState.REVOKED,
            )
        )


def test_transition_assignment_stale_revision_conflict() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    assignment = fixture.service.create_assignment(_create_assignment_request())
    fixture.service.transition_assignment(
        _transition_assignment_request(expected_revision=assignment.revision)
    )
    with pytest.raises(AssignmentRevisionConflict):
        fixture.service.transition_assignment(
            _transition_assignment_request(expected_revision=assignment.revision)
        )


def test_transition_assignment_denial_leaves_record_unchanged() -> None:
    fixture = _service_fixture()
    _create_work_item(fixture)
    assignment = fixture.service.create_assignment(_create_assignment_request())
    fixture.authority_repo.update(
        UpdatePrincipalAuthorityGrantCommand(
            scope=PrincipalAuthorityGrantScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                authority_grant_id="authority-grant-acting",
            ),
            expected_revision=0,
            authority_scopes=("collaborative_work.other",),
            status=AuthorityGrantStatus.ACTIVE,
        )
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.transition_assignment(
            _transition_assignment_request(expected_revision=assignment.revision)
        )
    unchanged = fixture.assignment_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        assignment_id="assignment-1",
    )
    assert unchanged == assignment


def test_transition_assignment_missing_raises_not_found() -> None:
    fixture = _service_fixture()
    with pytest.raises(AssignmentNotFound):
        fixture.service.transition_assignment(_transition_assignment_request())


# --- authority freshness ---


def test_authority_revoked_after_success_blocks_next_mutation() -> None:
    fixture = _service_fixture()
    created = _create_work_item(fixture)
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
        fixture.service.transition_work_item(
            _transition_work_item_request(expected_revision=created.revision)
        )


# --- delegation ---


def test_delegated_create_work_item_success() -> None:
    fixture = _service_fixture(seed_membership=False, seed_authority=False)
    _seed_delegated_memberships(fixture.membership_repo)
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
    _seed_authority(
        fixture.authority_repo,
        principal_id=_DELEGATOR,
        grant_id="authority-grant-delegator",
    )
    created = fixture.service.create_work_item(
        _create_work_item_request(
            membership_resolution_mode=MembershipResolutionMode.LOCATOR,
            membership=_acting_membership_locator(),
            delegator_principal_id=_DELEGATOR,
            delegation=_delegation_locator(resource_scope=resource_scope),
        )
    )
    assert created.work_item_id == _WORK_ITEM_ID


def test_delegated_mutation_denied_when_delegation_revoked() -> None:
    fixture = _service_fixture(seed_membership=False, seed_authority=False)
    _seed_delegated_memberships(fixture.membership_repo)
    resource_scope = work_item_resource_scope(work_item_id=_WORK_ITEM_ID)
    delegation = fixture.delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(_AUTHORITY_SCOPE,),
            resource_scope=resource_scope,
            status=DelegationStatus.REVOKED,
        )
    )
    _seed_authority(
        fixture.authority_repo,
        principal_id=_DELEGATOR,
        grant_id="authority-grant-delegator",
    )
    with pytest.raises(CollaborativeWorkAuthorizationDenied):
        fixture.service.create_work_item(
            _create_work_item_request(
                membership_resolution_mode=MembershipResolutionMode.LOCATOR,
                membership=_acting_membership_locator(),
                delegator_principal_id=_DELEGATOR,
                delegation=_delegation_locator(
                    resource_scope=resource_scope,
                    status=DelegationStatus.ACTIVE,
                ),
            )
        )
    assert delegation.status is DelegationStatus.REVOKED
