# © Artur Czarnecki. All rights reserved.

"""GR-10-R10-R1 — production DecisionRequirementPolicy resolution fail-closed proofs."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from intergrax.collaborative_work.in_memory_repository import (
    InMemoryAuthorityDelegationRepository,
    InMemoryCollaborativeOperationPolicyProfileRepository,
    InMemoryCollaborativePolicyRepository,
    InMemoryPrincipalAuthorityRepository,
    InMemoryWorkspaceMembershipRepository,
)
from intergrax.contracts.decision_requirement_policy import (
    DecisionRequirement,
    DecisionRequirementContext,
    DecisionRequirementPolicy,
)
from intergrax.contracts.meaningful_side_effect import (
    MeaningfulSideEffectKind,
    MeaningfulSideEffectRequest,
)
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.runtime.governance.decision_requirement_policy import (
    ConfiguredDecisionRequirementPolicy,
    DecisionRequirementRule,
    PermissiveDecisionRequirementPolicy,
    decision_governed_side_effect_requirement_policy,
)
from intergrax.runtime.governance.orchestration_decision_bound_effect_composition import (
    OrchestrationDecisionBoundCompositionError,
    build_production_orchestration_meaningful_side_effect_authorization_boundary,
    resolve_orchestration_decision_requirement_policy,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from testing_support.orchestration_governance_evidence_wiring import (
    default_test_orchestration_evidence_persistence,
)
from tests.unit.runtime.governance.gr3_test_support import (
    bound_gr3_active_execution,
    default_gr3_identity_bundle,
    default_gr3_inner_guard,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO = Path(__file__).resolve().parents[4]
_COMPOSITION = (
    _REPO
    / "intergrax"
    / "runtime"
    / "governance"
    / "orchestration_decision_bound_effect_composition.py"
)
_OPERATION = "orch.policy.proof"
_SCOPE = "orchestration/policy/proof"
_TENANT = "tenant-r10r1"


@dataclass(frozen=True, slots=True)
class _UndeterminedPolicy:
    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        return DecisionRequirement.UNDETERMINED


@dataclass(frozen=True, slots=True)
class _ExplodingPolicy:
    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        raise RuntimeError("policy exploded")


class _RecordingPolicy:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, context: DecisionRequirementContext) -> DecisionRequirement:
        self.calls += 1
        return DecisionRequirement.NOT_REQUIRED


def _repos_bundle() -> dict[str, object]:
    return {
        "profile_repository": InMemoryCollaborativeOperationPolicyProfileRepository(),
        "membership_repository": InMemoryWorkspaceMembershipRepository(),
        "principal_authority_repository": InMemoryPrincipalAuthorityRepository(),
        "delegation_repository": InMemoryAuthorityDelegationRepository(),
        "collaborative_policy_repository": InMemoryCollaborativePolicyRepository(),
        "runtime_policy_evaluator": RuntimePolicyEngine(),
    }


def test_resolve_production_missing_policy_raises() -> None:
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        resolve_orchestration_decision_requirement_policy(None, production_mode=True)


def test_resolve_non_production_missing_policy_uses_lab_permissive() -> None:
    policy = resolve_orchestration_decision_requirement_policy(None, production_mode=False)
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    ctx = DecisionRequirementContext(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
        action=_OPERATION,
        side_effect_scope_id=_SCOPE,
        resource_scope="resource",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert policy.evaluate(ctx) is DecisionRequirement.NOT_REQUIRED


def test_resolve_explicit_permissive_policy_accepted() -> None:
    explicit = PermissiveDecisionRequirementPolicy()
    resolved = resolve_orchestration_decision_requirement_policy(
        explicit,
        production_mode=True,
    )
    assert resolved is explicit


def test_build_production_boundary_missing_policy_composition_error() -> None:
    with pytest.raises(OrchestrationDecisionBoundCompositionError):
        build_production_orchestration_meaningful_side_effect_authorization_boundary(
            **_repos_bundle(),
            decision_requirement_policy=None,
            inner_execution_guard=default_gr3_inner_guard(default_gr3_identity_bundle()[0]),
            production_mode=True,
            governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
        )


def test_custom_protocol_policy_injected_without_concrete_dependency() -> None:
    custom = _RecordingPolicy()
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        **_repos_bundle(),
        decision_requirement_policy=custom,
        inner_execution_guard=default_gr3_inner_guard(default_gr3_identity_bundle()[0]),
        production_mode=True,
        governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
    )
    assert boundary._decision_requirement_policy is custom


def test_production_resolver_has_no_implicit_permissive_fallback_ast() -> None:
    source = _COMPOSITION.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(_COMPOSITION))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "resolve_orchestration_decision_requirement_policy":
            continue
        body_source = ast.get_source_segment(source, node) or ""
        assert "return PermissiveDecisionRequirementPolicy()" not in body_source
        assert "default_orchestration_decision_requirement_policy()" not in body_source
        return
    raise AssertionError("resolve_orchestration_decision_requirement_policy not found")


def test_undetermined_policy_zero_effect() -> None:
    from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
    from intergrax.runtime.policy.meaningful_side_effect_authorization import (
        MeaningfulSideEffectAuthorizationResult,
    )

    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        **_repos_bundle(),
        decision_requirement_policy=_UndeterminedPolicy(),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        production_mode=True,
        governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
    )
    side_effect = MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=_SCOPE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id="principal",
        tenant_id=_TENANT,
        resource="resource",
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id="ws",
        operation_id=_OPERATION,
        acting_principal_id="principal",
        resource_scope="resource",
        meaningful_side_effect_request=side_effect,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            request,
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_policy_exception_zero_effect() -> None:
    from unittest.mock import MagicMock

    from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
    from intergrax.runtime.policy.meaningful_side_effect_authorization import (
        MeaningfulSideEffectAuthorizationBoundary,
        MeaningfulSideEffectAuthorizationResult,
    )

    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=MagicMock(),
        inner_execution_guard=default_gr3_inner_guard(task_id),
        decision_requirement_policy=_ExplodingPolicy(),
    )
    side_effect = MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=_SCOPE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id="principal",
        tenant_id=_TENANT,
        resource="resource",
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id="ws",
        operation_id=_OPERATION,
        acting_principal_id="principal",
        resource_scope="resource",
        meaningful_side_effect_request=side_effect,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            request,
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert "policy evaluation failed" in (result.decision.reason or "")
    assert calls == []


def test_explicit_required_policy_blocks_missing_material() -> None:
    from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
    from intergrax.runtime.policy.meaningful_side_effect_authorization import (
        MeaningfulSideEffectAuthorizationResult,
    )

    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    policy = decision_governed_side_effect_requirement_policy(
        required_actions=frozenset({_OPERATION}),
    )
    boundary = build_production_orchestration_meaningful_side_effect_authorization_boundary(
        **_repos_bundle(),
        decision_requirement_policy=policy,
        inner_execution_guard=default_gr3_inner_guard(task_id),
        production_mode=True,
        governance_evidence_persistence=default_test_orchestration_evidence_persistence(),
    )
    side_effect = MeaningfulSideEffectRequest(
        action=_OPERATION,
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id=_SCOPE,
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        principal_id="principal",
        tenant_id=_TENANT,
        resource="resource",
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id=_TENANT,
        workspace_id="ws",
        operation_id=_OPERATION,
        acting_principal_id="principal",
        resource_scope="resource",
        meaningful_side_effect_request=side_effect,
    )
    calls: list[str] = []
    with bound_gr3_active_execution(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    ):
        result = boundary.authorize_and_execute(
            request,
            lambda: calls.append("effect") or "ok",
        )
    assert isinstance(result, MeaningfulSideEffectAuthorizationResult)
    assert result.decision.action is PolicyAction.DENY
    assert calls == []


def test_explicit_required_policy_classifies_required() -> None:
    policy = ConfiguredDecisionRequirementPolicy(
        rules=(
            DecisionRequirementRule(
                side_effect_scope_id=_SCOPE,
                action=_OPERATION,
            ),
        ),
    )
    task_id, run_id, attempt_id, execution_id = default_gr3_identity_bundle()
    ctx = DecisionRequirementContext(
        tenant_id=_TENANT,
        operation_id=_OPERATION,
        action=_OPERATION,
        side_effect_scope_id=_SCOPE,
        resource_scope="resource",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert policy.evaluate(ctx) is DecisionRequirement.REQUIRED
