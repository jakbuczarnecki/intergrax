# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1G — trusted enforcement gate tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.collaborative_work.authority import CollaborativeWorkAuthorityResolver
from intergrax.collaborative_work.enforcement_gate import CollaborativeWorkEnforcementGate
from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CreateCollaborativeOperationPolicyProfileCommand,
    CreateCollaborativePolicyRuleCommand,
    CreatePrincipalAuthorityGrantCommand,
    CreateWorkspaceMembershipCommand,
    UpdateWorkspaceMembershipCommand,
    WorkspaceMembershipScopeKey,
)
from intergrax.contracts.collaborative_work import (
    CollaborativeOperationPolicyProfileStatus,
    CollaborativeWorkEnforcementRequest,
    MembershipStatus,
    OperationPolicyRequirement,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
    WorkspaceMembership,
    WorkspaceMembershipRole,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_WORKSPACE = "workspace-a"
_ACTING = "principal-acting"
_OPERATION = "collaborative.document.delete"
_SCOPE = "document.delete"
_WEAKER_SCOPE = "document.read"
_RESOURCE = "document-123"
_NOW = datetime(2026, 6, 15, 12, 0, tzinfo=UTC)


def _membership_repo() -> InMemoryWorkspaceMembershipRepository:
    return InMemoryWorkspaceMembershipRepository()


def _authority_repo() -> InMemoryPrincipalAuthorityRepository:
    return InMemoryPrincipalAuthorityRepository()


def _policy_repo() -> InMemoryCollaborativePolicyRepository:
    return InMemoryCollaborativePolicyRepository()


def _profile_repo() -> InMemoryCollaborativeOperationPolicyProfileRepository:
    return InMemoryCollaborativeOperationPolicyProfileRepository()


def _seed_membership(repo: InMemoryWorkspaceMembershipRepository) -> WorkspaceMembership:
    return repo.create(
        CreateWorkspaceMembershipCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            membership_id="membership-1",
            principal_id=_ACTING,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.ACTIVE,
        )
    )


def _seed_authority(
    repo: InMemoryPrincipalAuthorityRepository,
    *,
    authority_scopes: tuple[str, ...] = (_SCOPE,),
) -> object:
    return repo.create(
        CreatePrincipalAuthorityGrantCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            authority_grant_id="authority-grant-1",
            principal_id=_ACTING,
            authority_scopes=authority_scopes,
        )
    )


def _create_profile_command(**overrides: object) -> CreateCollaborativeOperationPolicyProfileCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation_id": _OPERATION,
        "authority_scope": _SCOPE,
        "workspace_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "resource_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "runtime_policy_applicability": PolicyLayerApplicability.REQUIRED,
        "resource_requirement": OperationPolicyRequirement.REQUIRED,
        "meaningful_side_effect_requirement": OperationPolicyRequirement.REQUIRED,
        "status": CollaborativeOperationPolicyProfileStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativeOperationPolicyProfileCommand(**payload)


def _seed_workspace_allow(repo: InMemoryCollaborativePolicyRepository) -> None:
    repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="policy-ws-1",
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.ALLOW,
        )
    )


def _seed_resource_allow(repo: InMemoryCollaborativePolicyRepository) -> None:
    repo.create(
        CreateCollaborativePolicyRuleCommand(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id="policy-res-1",
            layer=PolicyCompositionLayer.RESOURCE_POLICY,
            authority_scope=_SCOPE,
            resource_scope=_RESOURCE,
            action=PolicyAction.ALLOW,
        )
    )


def _runtime_engine_allow() -> RuntimePolicyEngine:
    return RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": _OPERATION,
                "decision": "allow",
                "id": "runtime.allow",
            }
        ]
    )


def _runtime_engine_deny() -> RuntimePolicyEngine:
    return RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": _OPERATION,
                "decision": "deny",
                "id": "runtime.deny",
            }
        ]
    )


def _runtime_engine_require_human() -> RuntimePolicyEngine:
    return RuntimePolicyEngine(
        rules=[
            {
                "type": "meaningful_side_effect",
                "action": _OPERATION,
                "decision": "require_human",
                "id": "runtime.hitl",
            }
        ]
    )


def _membership_locator(membership: WorkspaceMembership) -> WorkspaceMembership:
    return WorkspaceMembership.model_validate(membership.model_dump())


def _runtime_request(**overrides: object) -> MeaningfulSideEffectRequest:
    payload = {
        "action": _OPERATION,
        "kinds": (MeaningfulSideEffectKind.MUTATION,),
        "task_id": "task-1",
        "run_id": "run-1",
        "principal_id": _ACTING,
        "tenant_id": _TENANT,
        "resource": _RESOURCE,
    }
    payload.update(overrides)
    return MeaningfulSideEffectRequest.model_validate(payload)


def _enforcement_request(**overrides: object) -> CollaborativeWorkEnforcementRequest:
    membership_repo = _membership_repo()
    membership = _seed_membership(membership_repo)
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "operation_id": _OPERATION,
        "acting_principal_id": _ACTING,
        "resource_scope": _RESOURCE,
        "membership": _membership_locator(membership),
        "meaningful_side_effect_request": _runtime_request(),
    }
    payload.update(overrides)
    return CollaborativeWorkEnforcementRequest.model_validate(payload)


def _gate(
    *,
    membership_repo: InMemoryWorkspaceMembershipRepository | None = None,
    authority_repo: InMemoryPrincipalAuthorityRepository | None = None,
    policy_repo: InMemoryCollaborativePolicyRepository | None = None,
    profile_repo: InMemoryCollaborativeOperationPolicyProfileRepository | None = None,
    runtime_engine: RuntimePolicyEngine | None = None,
) -> CollaborativeWorkEnforcementGate:
    membership = membership_repo or _membership_repo()
    if membership_repo is None:
        _seed_membership(membership)
    authority = authority_repo or _authority_repo()
    if authority_repo is None:
        _seed_authority(authority)
    policy = policy_repo or _policy_repo()
    profile = profile_repo or _profile_repo()
    if profile_repo is None:
        profile.create(_create_profile_command())
    return CollaborativeWorkEnforcementGate(
        profile_repository=profile,
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=membership,
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=authority,
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(policy),
        runtime_policy_evaluator=runtime_engine or _runtime_engine_allow(),
    )


# --- classification ---


def test_missing_operation_profile_denies() -> None:
    gate = _gate(profile_repo=_profile_repo())
    result = gate.evaluate(_enforcement_request())
    assert result.composition.decision.action is PolicyAction.DENY
    assert result.profile_revision is None


def test_disabled_profile_denies() -> None:
    profile_repo = _profile_repo()
    profile_repo.create(
        _create_profile_command(status=CollaborativeOperationPolicyProfileStatus.DISABLED)
    )
    result = _gate(profile_repo=profile_repo).evaluate(_enforcement_request())
    assert result.composition.decision.action is PolicyAction.DENY


def test_caller_cannot_supply_applicability_override() -> None:
    assert "applicability" not in CollaborativeWorkEnforcementRequest.model_fields


def test_profile_unknown_applicability_cannot_be_authored() -> None:
    profile_repo = _profile_repo()
    with pytest.raises(ValueError, match="UNKNOWN"):
        profile_repo.create(
            _create_profile_command(workspace_policy_applicability=PolicyLayerApplicability.UNKNOWN)
        )


# --- authority ---


def test_profile_authority_scope_controls_authority_evaluation() -> None:
    authority_repo = _authority_repo()
    _seed_authority(authority_repo, authority_scopes=(_WEAKER_SCOPE,))
    profile_repo = _profile_repo()
    profile_repo.create(_create_profile_command(authority_scope=_SCOPE))
    result = _gate(authority_repo=authority_repo, profile_repo=profile_repo).evaluate(
        _enforcement_request()
    )
    assert result.composition.collaborative_authority.action is PolicyAction.DENY
    assert result.authority_scope == _SCOPE


def test_caller_cannot_substitute_weaker_authority_scope() -> None:
    authority_repo = _authority_repo()
    _seed_authority(authority_repo, authority_scopes=(_WEAKER_SCOPE,))
    profile_repo = _profile_repo()
    profile_repo.create(_create_profile_command(authority_scope=_SCOPE))
    result = _gate(authority_repo=authority_repo, profile_repo=profile_repo).evaluate(
        _enforcement_request()
    )
    assert result.composition.decision.action is PolicyAction.DENY


def test_collaborative_deny_survives_final_gate() -> None:
    membership_repo = _membership_repo()
    membership = _seed_membership(membership_repo)
    membership_repo.update(
        UpdateWorkspaceMembershipCommand(
            scope=WorkspaceMembershipScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                membership_id="membership-1",
            ),
            expected_revision=membership.revision,
            role=WorkspaceMembershipRole.MEMBER,
            status=MembershipStatus.REVOKED,
        )
    )
    result = _gate(membership_repo=membership_repo).evaluate(
        _enforcement_request(membership=_membership_locator(membership))
    )
    assert result.composition.decision.action is PolicyAction.DENY


# --- resource ---


def test_resource_required_operation_missing_resource_denies() -> None:
    result = _gate().evaluate(_enforcement_request(resource_scope=None))
    assert result.composition.decision.action is PolicyAction.DENY


def test_wrong_resource_policy_rule_denies() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo).evaluate(
        _enforcement_request(resource_scope="other-resource")
    )
    assert result.composition.decision.action is PolicyAction.DENY


def test_resource_not_applicable_profile_skips_resource_evaluator() -> None:
    profile_repo = _profile_repo()
    profile_repo.create(
        _create_profile_command(
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
        )
    )
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    result = _gate(profile_repo=profile_repo, policy_repo=policy_repo).evaluate(
        _enforcement_request(
            resource_scope=None,
            meaningful_side_effect_request=None,
        )
    )
    assert result.composition.resource_policy is None
    assert result.composition.decision.action is PolicyAction.ALLOW


# --- workspace ---


def test_required_workspace_rule_allow_participates() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo).evaluate(_enforcement_request())
    assert result.composition.workspace_policy is not None
    assert result.composition.workspace_policy.action is PolicyAction.ALLOW


def test_missing_workspace_rule_denies() -> None:
    policy_repo = _policy_repo()
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo).evaluate(_enforcement_request())
    assert result.composition.decision.action is PolicyAction.DENY


# --- runtime ---


def test_meaningful_side_effect_required_missing_runtime_request_denies() -> None:
    result = _gate().evaluate(_enforcement_request(meaningful_side_effect_request=None))
    assert result.composition.decision.action is PolicyAction.DENY


def test_runtime_identity_mismatch_denies() -> None:
    result = _gate().evaluate(
        _enforcement_request(
            meaningful_side_effect_request=_runtime_request(
                action="collaborative.other.operation",
            )
        )
    )
    assert result.composition.decision.action is PolicyAction.DENY


def test_runtime_deny_survives() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo, runtime_engine=_runtime_engine_deny()).evaluate(
        _enforcement_request()
    )
    assert result.composition.decision.action is PolicyAction.DENY
    assert result.composition.runtime_policy is not None
    assert result.composition.runtime_policy.action is PolicyAction.DENY


def test_runtime_require_human_survives() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(
        policy_repo=policy_repo,
        runtime_engine=_runtime_engine_require_human(),
    ).evaluate(_enforcement_request())
    assert result.composition.decision.action is PolicyAction.REQUIRE_HUMAN


def test_runtime_allow_participates_normally() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo).evaluate(_enforcement_request())
    assert result.composition.runtime_policy is not None
    assert result.composition.runtime_policy.action is PolicyAction.ALLOW


# --- final ---


def test_all_required_layers_allow_yields_final_allow() -> None:
    policy_repo = _policy_repo()
    _seed_workspace_allow(policy_repo)
    _seed_resource_allow(policy_repo)
    result = _gate(policy_repo=policy_repo).evaluate(_enforcement_request())
    assert result.composition.decision.action is PolicyAction.ALLOW
    assert result.profile_revision == 0
    assert result.authority_scope == _SCOPE


def test_authoritative_not_applicable_layers_skipped() -> None:
    profile_repo = _profile_repo()
    profile_repo.create(
        _create_profile_command(
            workspace_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            runtime_policy_applicability=PolicyLayerApplicability.NOT_APPLICABLE,
            resource_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
            meaningful_side_effect_requirement=OperationPolicyRequirement.NOT_APPLICABLE,
        )
    )
    result = _gate(profile_repo=profile_repo).evaluate(
        _enforcement_request(
            resource_scope=None,
            meaningful_side_effect_request=None,
        )
    )
    assert result.composition.decision.action is PolicyAction.ALLOW
    assert result.composition.workspace_policy is None
    assert result.composition.resource_policy is None
    assert result.composition.runtime_policy is None


def test_caller_cannot_create_final_allow_by_omitting_policy_inputs() -> None:
    profile_repo = _profile_repo()
    profile_repo.create(_create_profile_command())
    membership_repo = _membership_repo()
    membership = _seed_membership(membership_repo)
    gate = _gate(profile_repo=profile_repo, membership_repo=membership_repo)
    result = gate.evaluate(
        CollaborativeWorkEnforcementRequest(
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            operation_id=_OPERATION,
            acting_principal_id=_ACTING,
            membership=_membership_locator(membership),
        )
    )
    assert result.composition.decision.action is PolicyAction.DENY
