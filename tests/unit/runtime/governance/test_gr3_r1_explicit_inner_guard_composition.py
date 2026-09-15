# © Artur Czarnecki. All rights reserved.

"""GR-3-R1 — explicit inner guard composition and contract-only consumer."""

from __future__ import annotations

import inspect
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
from intergrax.contracts.canonical_inner_governance import (
    CanonicalInnerExecutionGuardPort,
    CanonicalInnerGovernanceViolation,
)
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import mint_task_id
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind, MeaningfulSideEffectRequest
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_canonical_inner_execution_guard,
    build_default_wired_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationBoundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine

pytestmark = pytest.mark.unit

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_TASK = mint_task_id()


def _minimal_gate() -> CollaborativeWorkEnforcementGate:
    return CollaborativeWorkEnforcementGate(
        profile_repository=InMemoryCollaborativeOperationPolicyProfileRepository(),
        authority_resolver=CollaborativeWorkAuthorityResolver(
            membership_repository=InMemoryWorkspaceMembershipRepository(),
            delegation_repository=InMemoryAuthorityDelegationRepository(),
            principal_authority_repository=InMemoryPrincipalAuthorityRepository(),
            clock=lambda: _NOW,
        ),
        policy_evaluator=CollaborativePolicyEvaluator(InMemoryCollaborativePolicyRepository()),
        runtime_policy_evaluator=RuntimePolicyEngine(),
    )


def _minimal_side_effect() -> MeaningfulSideEffectRequest:
    from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id

    return MeaningfulSideEffectRequest(
        action="op",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope",
        task_id=_TASK,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        principal_id="principal",
        tenant_id="tenant",
        resource="resource",
    )


class _AllowingTestGuard(CanonicalInnerExecutionGuardPort):
    def assert_meaningful_side_effect_bound(self, request: MeaningfulSideEffectRequest) -> None:
        del request


class _RejectingTestGuard(CanonicalInnerExecutionGuardPort):
    def assert_meaningful_side_effect_bound(self, request: MeaningfulSideEffectRequest) -> None:
        del request
        raise CanonicalInnerGovernanceViolation(reason="test-reject")


def test_boundary_constructor_requires_inner_execution_guard() -> None:
    params = inspect.signature(MeaningfulSideEffectAuthorizationBoundary.__init__).parameters
    guard_param = params["inner_execution_guard"]
    assert guard_param.default is inspect.Parameter.empty


def test_boundary_rejects_missing_inner_guard_at_runtime() -> None:
    gate = _minimal_gate()
    with pytest.raises(TypeError):
        MeaningfulSideEffectAuthorizationBoundary(enforcement_gate=gate)  # type: ignore[call-arg]


def test_custom_guard_is_invoked_and_can_block() -> None:
    gate = _minimal_gate()
    boundary = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=_RejectingTestGuard(),
    )
    request = CollaborativeWorkEnforcementRequest(
        tenant_id="tenant",
        workspace_id="ws",
        operation_id="op",
        acting_principal_id="principal",
        resource_scope="resource",
        membership=None,
        meaningful_side_effect_request=_minimal_side_effect(),
    )
    result = boundary.authorize(request)
    assert result.permitted is False
    assert result.decision.policy_rule_id == "platform.canonical_inner_enforcement"


def test_two_guards_same_consumer_no_source_branching() -> None:
    gate = _minimal_gate()
    boundary_a = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=_AllowingTestGuard(),
    )
    boundary_b = MeaningfulSideEffectAuthorizationBoundary(
        enforcement_gate=gate,
        inner_execution_guard=_RejectingTestGuard(),
    )
    assert type(boundary_a) is type(boundary_b)
    assert boundary_a.__class__ is boundary_b.__class__


def test_default_composition_builder_wires_platform_guard() -> None:
    gate = _minimal_gate()
    boundary = build_default_wired_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
    )
    guard = build_default_canonical_inner_execution_guard()
    assert isinstance(boundary, MeaningfulSideEffectAuthorizationBoundary)
    assert hasattr(guard, "assert_meaningful_side_effect_bound")
