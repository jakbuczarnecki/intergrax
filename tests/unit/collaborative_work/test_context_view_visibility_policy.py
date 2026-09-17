# © Artur Czarnecki. All rights reserved.

"""MP-5C — ContextView visibility policy tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.context_view_visibility import (
    ContextViewVisibilityEvaluator,
    DefaultContextViewVisibilityPolicy,
)
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.repository import (
    CreateAuthorityDelegationCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
)
from intergrax.contracts.collaborative_work import (
    MembershipResolutionMode,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.context_view import (
    ContextViewCategory,
    ContextViewOperationScope,
    ContextViewRequest,
    ContextViewScope,
    ContextViewVisibilityClass,
)
from intergrax.contracts.context_view_visibility_policy import (
    CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
    ContextViewCategoryDenialReason,
    ContextViewPolicyDenialReason,
    ContextViewPolicyDecision,
    ContextViewPolicyOutcome,
    ContextViewVisibilityPolicy,
    ContextViewVisibilityPolicyInput,
    DefaultContextViewVisibilityPolicyConfig,
    fail_closed_context_view_policy_decision,
)

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_ACTING = "principal-acting"
_OTHER = "principal-other"
_DELEGATOR = "principal-delegator"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)
_OP = "collaborative_work.context_view.compose"


def _scope(**overrides: object) -> ContextViewScope:
    payload = {"tenant_id": _TENANT, "workspace_id": _WORKSPACE}
    payload.update(overrides)
    return ContextViewScope(**payload)


def _request(**overrides: object) -> ContextViewRequest:
    payload = {
        "scope": _scope(),
        "acting_principal_id": _ACTING,
        "operation_id": _OP,
        "requested_categories": (ContextViewCategory.MEMORY,),
        "membership_resolution_mode": MembershipResolutionMode.CANONICAL_PRINCIPAL,
    }
    payload.update(overrides)
    return ContextViewRequest(**payload)


def _membership_repo() -> InMemoryWorkspaceMembershipRepository:
    return InMemoryWorkspaceMembershipRepository()


def _delegation_repo() -> InMemoryAuthorityDelegationRepository:
    return InMemoryAuthorityDelegationRepository()


def _authority_repo() -> InMemoryPrincipalAuthorityRepository:
    return InMemoryPrincipalAuthorityRepository()


def _resolver(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    delegation_repo: InMemoryAuthorityDelegationRepository | None = None,
    authority_repo: InMemoryPrincipalAuthorityRepository | None = None,
) -> CollaborativeWorkAuthorityResolver:
    return CollaborativeWorkAuthorityResolver(
        membership_repository=membership_repo or _membership_repo(),
        delegation_repository=delegation_repo or _delegation_repo(),
        principal_authority_repository=authority_repo or _authority_repo(),
        clock=lambda: _NOW,
    )


def _seed_membership(
    repo: InMemoryWorkspaceMembershipRepository,
    *,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    principal_id: str = _ACTING,
) -> WorkspaceMembership:
    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            membership_id=f"membership-{principal_id}",
            principal_id=principal_id,
            role=WorkspaceMembershipRole.MEMBER,
        )
    )


def _seed_authority(
    repo: InMemoryPrincipalAuthorityRepository,
    *,
    tenant_id: str = _TENANT,
    workspace_id: str = _WORKSPACE,
    principal_id: str = _ACTING,
    authority_scopes: tuple[str, ...] = (CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
) -> object:
    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            authority_grant_id=f"grant-{principal_id}",
            principal_id=principal_id,
            authority_scopes=authority_scopes,
        )
    )


def _evaluator(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    delegation_repo: InMemoryAuthorityDelegationRepository | None = None,
    authority_repo: InMemoryPrincipalAuthorityRepository | None = None,
    policy: ContextViewVisibilityPolicy | None = None,
    config: DefaultContextViewVisibilityPolicyConfig | None = None,
) -> ContextViewVisibilityEvaluator:
    membership_repo = membership_repo or _membership_repo()
    delegation_repo = delegation_repo or _delegation_repo()
    authority_repo = authority_repo or _authority_repo()
    policy = policy or DefaultContextViewVisibilityPolicy(config)
    return ContextViewVisibilityEvaluator(
        authority_resolver=_resolver(
            membership_repo=membership_repo,
            delegation_repo=delegation_repo,
            authority_repo=authority_repo,
        ),
        visibility_policy=policy,
        delegation_repository=delegation_repo,
        policy_config=config,
    )


class _StubVisibilityPolicy:
    def __init__(self, *, policy_id: str = "stub.policy") -> None:
        self._policy_id = policy_id
        self.calls: list[ContextViewVisibilityPolicyInput] = []

    @property
    def policy_id(self) -> str:
        return self._policy_id

    def evaluate(self, policy_input: ContextViewVisibilityPolicyInput) -> ContextViewPolicyDecision:
        self.calls.append(policy_input)
        return fail_closed_context_view_policy_decision(
            policy_id=self.policy_id,
            effective_scope=policy_input.request.scope,
            denial_reason=ContextViewPolicyDenialReason.POLICY_AMBIGUITY,
        )


def test_authorized_principal_eligible_for_requested_category() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(_request())
    assert decision.outcome is ContextViewPolicyOutcome.ALLOW
    assert ContextViewCategory.MEMORY in decision.eligible_categories


def test_workspace_member_workspace_shared_allowed() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(
        _request(
            requested_categories=(ContextViewCategory.KNOWLEDGE,),
        ),
    )
    assert ContextViewVisibilityClass.WORKSPACE_SHARED in decision.eligible_visibility_classes


def test_work_item_scoped_request_allows_work_item_visibility() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(
        _request(
            scope=_scope(work_item_id="wi-1"),
            requested_categories=(ContextViewCategory.COLLABORATIVE_WORK,),
        ),
    )
    assert ContextViewVisibilityClass.WORK_ITEM in decision.eligible_visibility_classes


def test_partial_category_eligibility_subset() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo, principal_id=_ACTING)
    _seed_membership(membership_repo, principal_id=_DELEGATOR)
    _seed_authority(authority_repo, principal_id=_DELEGATOR)
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-1",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
        )
    )
    locator = delegation_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        delegation_id="delegation-1",
    )
    assert locator is not None
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    ).evaluate(
        _request(
            acting_principal_id=_ACTING,
            delegator_principal_id=_DELEGATOR,
            delegation=locator,
            requested_categories=(
                ContextViewCategory.MEMORY,
                ContextViewCategory.KNOWLEDGE,
            ),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.ALLOW
    assert ContextViewCategory.MEMORY in decision.eligible_categories
    assert ContextViewCategory.KNOWLEDGE not in decision.eligible_categories
    assert any(
        denial.category is ContextViewCategory.KNOWLEDGE
        and denial.reason is ContextViewCategoryDenialReason.DELEGATION_SCOPE_INSUFFICIENT
        for denial in decision.denied_categories
    )


def test_missing_membership_denies() -> None:
    authority_repo = _authority_repo()
    _seed_authority(authority_repo)
    decision = _evaluator(authority_repo=authority_repo).evaluate(_request())
    assert decision.outcome is ContextViewPolicyOutcome.DENY
    assert decision.denial_reason is ContextViewPolicyDenialReason.AUTHORITY_DENIED


def test_missing_authority_grant_denies() -> None:
    membership_repo = _membership_repo()
    _seed_membership(membership_repo)
    decision = _evaluator(membership_repo=membership_repo).evaluate(_request())
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_wrong_tenant_denies() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo, tenant_id=_TENANT)
    _seed_authority(authority_repo, tenant_id=_TENANT)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(
        _request(
            scope=_scope(tenant_id=_OTHER_TENANT),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_wrong_workspace_denies() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo, workspace_id=_WORKSPACE)
    _seed_authority(authority_repo, workspace_id=_WORKSPACE)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(
        _request(
            scope=_scope(workspace_id=_OTHER_WORKSPACE),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_delegation_outside_resource_scope_denies() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo, principal_id=_ACTING)
    _seed_membership(membership_repo, principal_id=_DELEGATOR)
    _seed_authority(authority_repo, principal_id=_DELEGATOR)
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-resource",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(
                CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
                "collaborative_work.context_view.workspace_shared",
            ),
            resource_scope="resource-a",
        )
    )
    locator = delegation_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        delegation_id="delegation-resource",
    )
    assert locator is not None
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    ).evaluate(
        _request(
            delegator_principal_id=_DELEGATOR,
            delegation=locator,
            scope=_scope(
                operation_scope=ContextViewOperationScope(
                    operation_id=_OP,
                    resource_scope="resource-b",
                ),
            ),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_delegation_amplification_denied_for_workspace_shared() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    delegation_repo = _delegation_repo()
    _seed_membership(membership_repo, principal_id=_ACTING)
    _seed_membership(membership_repo, principal_id=_DELEGATOR)
    _seed_authority(
        authority_repo,
        principal_id=_DELEGATOR,
        authority_scopes=(
            CONTEXT_VIEW_READ_AUTHORITY_SCOPE,
            "collaborative_work.context_view.workspace_shared",
        ),
    )
    delegation_repo.create(
        CreateAuthorityDelegationCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            delegation_id="delegation-narrow",
            delegator_principal_id=_DELEGATOR,
            delegate_principal_id=_ACTING,
            authority_scopes=(CONTEXT_VIEW_READ_AUTHORITY_SCOPE,),
        )
    )
    locator = delegation_repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        delegation_id="delegation-narrow",
    )
    assert locator is not None
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        delegation_repo=delegation_repo,
    ).evaluate(
        _request(
            delegator_principal_id=_DELEGATOR,
            delegation=locator,
            requested_categories=(ContextViewCategory.KNOWLEDGE,),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY
    assert decision.denial_reason is ContextViewPolicyDenialReason.NO_ELIGIBLE_CATEGORIES


def test_private_visibility_scoped_to_acting_principal_only() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo, principal_id=_ACTING)
    _seed_membership(membership_repo, principal_id=_OTHER)
    _seed_authority(authority_repo, principal_id=_ACTING)
    _seed_authority(authority_repo, principal_id=_OTHER)
    evaluator = _evaluator(membership_repo=membership_repo, authority_repo=authority_repo)
    decision_a = evaluator.evaluate(_request(acting_principal_id=_ACTING))
    decision_b = evaluator.evaluate(_request(acting_principal_id=_OTHER))
    assert decision_a.private_visibility_principal_id == _ACTING
    assert decision_b.private_visibility_principal_id == _OTHER
    assert decision_a.private_visibility_principal_id != decision_b.private_visibility_principal_id


def test_workspace_shared_requires_membership_not_bypassed() -> None:
    decision = _evaluator().evaluate(
        _request(requested_categories=(ContextViewCategory.KNOWLEDGE,)),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_delegated_visible_requires_canonical_delegation() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo, principal_id=_ACTING)
    _seed_authority(authority_repo, principal_id=_ACTING)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(
        _request(
            delegator_principal_id=_DELEGATOR,
            requested_categories=(ContextViewCategory.MEMORY,),
        ),
    )
    assert decision.outcome is ContextViewPolicyOutcome.DENY


def test_platform_visible_still_requires_tenant_scope() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    decision = _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
    ).evaluate(_request())
    assert ContextViewVisibilityClass.PLATFORM_VISIBLE in decision.eligible_visibility_classes
    assert decision.effective_scope.tenant_id == _TENANT


def test_policy_deterministic_for_same_inputs() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    evaluator = _evaluator(membership_repo=membership_repo, authority_repo=authority_repo)
    request = _request(
        requested_categories=(
            ContextViewCategory.MEMORY,
            ContextViewCategory.KNOWLEDGE,
        ),
    )
    first = evaluator.evaluate(request)
    second = evaluator.evaluate(request)
    assert first == second


def test_custom_policy_replaceable_without_abi_change() -> None:
    membership_repo = _membership_repo()
    authority_repo = _authority_repo()
    _seed_membership(membership_repo)
    _seed_authority(authority_repo)
    stub = _StubVisibilityPolicy(policy_id="custom.visibility.policy")
    _evaluator(
        membership_repo=membership_repo,
        authority_repo=authority_repo,
        policy=stub,
    ).evaluate(_request())
    assert len(stub.calls) == 1
    assert stub.policy_id == "custom.visibility.policy"
