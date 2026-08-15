# © Artur Czarnecki. All rights reserved.

"""COLLAB-WORK-1F — workspace/resource policy source and evaluator tests."""

from __future__ import annotations

import pytest

from intergrax.collaborative_work.in_memory_repository import InMemoryCollaborativePolicyRepository
from intergrax.collaborative_work.policy_composition import compose_policy_decisions
from intergrax.collaborative_work.policy_source import CollaborativePolicyEvaluator
from intergrax.collaborative_work.repository import (
    CollaborativePolicyRuleAlreadyExists,
    CollaborativePolicyRuleIdempotencyConflict,
    CollaborativePolicyRuleNotFound,
    CollaborativePolicyRuleRevisionConflict,
    CollaborativePolicyRuleScopeKey,
    CreateCollaborativePolicyRuleCommand,
    INITIAL_RECORD_REVISION,
    UpdateCollaborativePolicyRuleCommand,
)
from intergrax.contracts.collaborative_work import (
    CollaborativePolicyRule,
    CollaborativePolicyRuleStatus,
    PolicyCompositionApplicability,
    PolicyCompositionInput,
    PolicyCompositionLayer,
    PolicyLayerApplicability,
)
from intergrax.contracts.runtime_policy import PolicyAction, PolicyDecision

pytestmark = pytest.mark.unit

_TENANT = "tenant-a"
_OTHER_TENANT = "tenant-b"
_WORKSPACE = "workspace-a"
_OTHER_WORKSPACE = "workspace-b"
_SCOPE = "workspace.delete"
_RESOURCE = "document-123"
_RULE_WS = "policy-ws-1"
_RULE_RES = "policy-res-1"


def _repo() -> InMemoryCollaborativePolicyRepository:
    return InMemoryCollaborativePolicyRepository()


def _evaluator(repo: InMemoryCollaborativePolicyRepository | None = None) -> CollaborativePolicyEvaluator:
    return CollaborativePolicyEvaluator(repo or _repo())


def _create_workspace_rule_command(**overrides: object) -> CreateCollaborativePolicyRuleCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "policy_rule_id": _RULE_WS,
        "layer": PolicyCompositionLayer.WORKSPACE_POLICY,
        "authority_scope": _SCOPE,
        "action": PolicyAction.ALLOW,
        "status": CollaborativePolicyRuleStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativePolicyRuleCommand(**payload)


def _create_resource_rule_command(**overrides: object) -> CreateCollaborativePolicyRuleCommand:
    payload = {
        "tenant_id": _TENANT,
        "workspace_id": _WORKSPACE,
        "policy_rule_id": _RULE_RES,
        "layer": PolicyCompositionLayer.RESOURCE_POLICY,
        "authority_scope": _SCOPE,
        "resource_scope": _RESOURCE,
        "action": PolicyAction.REQUIRE_HUMAN,
        "status": CollaborativePolicyRuleStatus.ACTIVE,
    }
    payload.update(overrides)
    return CreateCollaborativePolicyRuleCommand(**payload)


def _allow(rule_id: str = "collab.allow") -> PolicyDecision:
    return PolicyDecision(action=PolicyAction.ALLOW, reason="allow", policy_rule_id=rule_id)


def _all_required_applicability() -> PolicyCompositionApplicability:
    return PolicyCompositionApplicability(
        workspace_policy=PolicyLayerApplicability.REQUIRED,
        resource_policy=PolicyLayerApplicability.REQUIRED,
        runtime_policy=PolicyLayerApplicability.REQUIRED,
    )


# --- workspace evaluation ---


def test_workspace_active_allow_rule_returns_allow() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(action=PolicyAction.ALLOW))
    decision = _evaluator(repo).evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.ALLOW
    assert decision.policy_rule_id == _RULE_WS


def test_workspace_deny_rule_returns_deny() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(action=PolicyAction.DENY))
    decision = _evaluator(repo).evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY


def test_workspace_require_human_preserved() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(action=PolicyAction.REQUIRE_HUMAN))
    decision = _evaluator(repo).evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.REQUIRE_HUMAN


def test_workspace_missing_rule_denies() -> None:
    decision = _evaluator().evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "collaborative_work.workspace_policy.missing"


def test_workspace_disabled_rule_denies() -> None:
    repo = _repo()
    repo.create(
        _create_workspace_rule_command(status=CollaborativePolicyRuleStatus.DISABLED),
    )
    decision = _evaluator(repo).evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY
    assert decision.policy_rule_id == "collaborative_work.workspace_policy.inactive"


def test_workspace_foreign_tenant_or_workspace_cannot_match() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command())
    evaluator = _evaluator(repo)
    assert evaluator.evaluate_workspace_policy(
        tenant_id=_OTHER_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    ).action is PolicyAction.DENY
    assert evaluator.evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_OTHER_WORKSPACE,
        authority_scope=_SCOPE,
    ).action is PolicyAction.DENY


def test_workspace_wrong_authority_scope_denies() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command())
    decision = _evaluator(repo).evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope="workspace.read",
    )
    assert decision.action is PolicyAction.DENY


# --- resource evaluation ---


def test_resource_exact_match_returns_configured_action() -> None:
    repo = _repo()
    repo.create(_create_resource_rule_command())
    decision = _evaluator(repo).evaluate_resource_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        resource_scope=_RESOURCE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.REQUIRE_HUMAN
    assert decision.policy_rule_id == _RULE_RES


def test_resource_wrong_resource_denies() -> None:
    repo = _repo()
    repo.create(_create_resource_rule_command())
    decision = _evaluator(repo).evaluate_resource_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        resource_scope="document-999",
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY


def test_resource_missing_rule_denies() -> None:
    decision = _evaluator().evaluate_resource_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        resource_scope=_RESOURCE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY


def test_workspace_rule_does_not_substitute_resource_rule() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(action=PolicyAction.ALLOW))
    decision = _evaluator(repo).evaluate_resource_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        resource_scope=_RESOURCE,
        authority_scope=_SCOPE,
    )
    assert decision.action is PolicyAction.DENY


# --- repository ---


def test_repository_duplicate_exact_policy_key_rejected() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(policy_rule_id="rule-a"))
    with pytest.raises(CollaborativePolicyRuleAlreadyExists):
        repo.create(
            _create_workspace_rule_command(
                policy_rule_id="rule-b",
                idempotency_key=None,
            )
        )


def test_repository_revision_increment() -> None:
    repo = _repo()
    created = repo.create(_create_workspace_rule_command())
    updated = repo.update(
        UpdateCollaborativePolicyRuleCommand(
            scope=CollaborativePolicyRuleScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                policy_rule_id=_RULE_WS,
            ),
            expected_revision=INITIAL_RECORD_REVISION,
            action=PolicyAction.DENY,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    assert created.revision == INITIAL_RECORD_REVISION
    assert updated.revision == INITIAL_RECORD_REVISION + 1


def test_repository_stale_revision_conflict_preserves_state() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command())
    with pytest.raises(CollaborativePolicyRuleRevisionConflict):
        repo.update(
            UpdateCollaborativePolicyRuleCommand(
                scope=CollaborativePolicyRuleScopeKey(
                    tenant_id=_TENANT,
                    workspace_id=_WORKSPACE,
                    policy_rule_id=_RULE_WS,
                ),
                expected_revision=99,
                action=PolicyAction.DENY,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )
    current = repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        policy_rule_id=_RULE_WS,
    )
    assert current is not None
    assert current.action is PolicyAction.ALLOW
    assert current.revision == INITIAL_RECORD_REVISION


def test_repository_scoped_isolation() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command())
    assert (
        repo.get(
            tenant_id=_OTHER_TENANT,
            workspace_id=_WORKSPACE,
            policy_rule_id=_RULE_WS,
        )
        is None
    )
    with pytest.raises(CollaborativePolicyRuleNotFound):
        repo.update(
            UpdateCollaborativePolicyRuleCommand(
                scope=CollaborativePolicyRuleScopeKey(
                    tenant_id=_OTHER_TENANT,
                    workspace_id=_WORKSPACE,
                    policy_rule_id=_RULE_WS,
                ),
                expected_revision=INITIAL_RECORD_REVISION,
                action=PolicyAction.DENY,
                status=CollaborativePolicyRuleStatus.ACTIVE,
            )
        )


def test_repository_idempotent_replay() -> None:
    repo = _repo()
    command = _create_workspace_rule_command(idempotency_key="idem-1")
    first = repo.create(command)
    second = repo.create(command)
    assert first == second


def test_repository_delayed_replay_after_update_returns_original_create_snapshot() -> None:
    repo = _repo()
    command = _create_workspace_rule_command(idempotency_key="idem-2")
    original = repo.create(command)
    repo.update(
        UpdateCollaborativePolicyRuleCommand(
            scope=CollaborativePolicyRuleScopeKey(
                tenant_id=_TENANT,
                workspace_id=_WORKSPACE,
                policy_rule_id=_RULE_WS,
            ),
            expected_revision=INITIAL_RECORD_REVISION,
            action=PolicyAction.DENY,
            status=CollaborativePolicyRuleStatus.ACTIVE,
        )
    )
    replay = repo.create(command)
    assert replay == original
    current = repo.get(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        policy_rule_id=_RULE_WS,
    )
    assert current is not None
    assert current.action is PolicyAction.DENY
    assert current.revision == INITIAL_RECORD_REVISION + 1


def test_repository_idempotency_conflict() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(idempotency_key="idem-3"))
    with pytest.raises(CollaborativePolicyRuleIdempotencyConflict):
        repo.create(
            _create_workspace_rule_command(
                action=PolicyAction.DENY,
                idempotency_key="idem-3",
            )
        )


def test_contract_rejects_modify_action() -> None:
    with pytest.raises(ValueError, match="MODIFY"):
        CollaborativePolicyRule(
            policy_rule_id=_RULE_WS,
            tenant_id=_TENANT,
            workspace_id=_WORKSPACE,
            layer=PolicyCompositionLayer.WORKSPACE_POLICY,
            authority_scope=_SCOPE,
            action=PolicyAction.MODIFY,
            revision=0,
        )


# --- composition integration ---


def test_evaluator_outputs_compose_with_policy_composition() -> None:
    repo = _repo()
    repo.create(_create_workspace_rule_command(action=PolicyAction.ALLOW))
    repo.create(_create_resource_rule_command(action=PolicyAction.REQUIRE_HUMAN))
    evaluator = _evaluator(repo)
    workspace = evaluator.evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    resource = evaluator.evaluate_resource_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        resource_scope=_RESOURCE,
        authority_scope=_SCOPE,
    )
    result = compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=_allow(),
            workspace_policy=workspace,
            resource_policy=resource,
            runtime_policy=_allow("runtime.allow"),
            applicability=_all_required_applicability(),
        )
    )
    assert result.decision.action is PolicyAction.REQUIRE_HUMAN
    assert result.determining_layer is PolicyCompositionLayer.RESOURCE_POLICY


def test_missing_policy_remains_restrictive_in_composition() -> None:
    evaluator = _evaluator()
    workspace = evaluator.evaluate_workspace_policy(
        tenant_id=_TENANT,
        workspace_id=_WORKSPACE,
        authority_scope=_SCOPE,
    )
    result = compose_policy_decisions(
        PolicyCompositionInput(
            collaborative_authority=_allow(),
            workspace_policy=workspace,
            applicability=_all_required_applicability(),
        )
    )
    assert result.decision.action is PolicyAction.DENY


def test_no_runtime_policy_engine_duplication() -> None:
    """Evaluator depends on repository port only — no RuntimePolicyEngine import."""
    import intergrax.collaborative_work.policy_source as module

    source = module.__file__
    assert source is not None
    with open(source, encoding="utf-8") as handle:
        contents = handle.read()
    assert "RuntimePolicyEngine" not in contents
    assert "PolicyEngine" not in contents
