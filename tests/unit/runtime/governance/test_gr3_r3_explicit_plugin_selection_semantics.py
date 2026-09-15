# © Artur Czarnecki. All rights reserved.

"""GR-3-R3 — explicit plugin selection (None only, not truthiness)."""

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
from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
from intergrax.contracts.collaborative_work import CollaborativeWorkEnforcementRequest
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.meaningful_side_effect import MeaningfulSideEffectKind, MeaningfulSideEffectRequest
from intergrax.runtime.governance.canonical_inner_execution_guard import (
    DefaultCanonicalInnerExecutionGuard,
)
from intergrax.runtime.governance.meaningful_side_effect_authorization_composition import (
    build_default_wired_meaningful_side_effect_authorization_boundary,
)
from intergrax.runtime.policy.runtime_policy_engine import RuntimePolicyEngine
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver
from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution

pytestmark = pytest.mark.unit

_NOW = datetime(2025, 1, 1, tzinfo=UTC)
_TASK = mint_task_id()


class _FalseyTaskScope(ActiveExecutionTaskScopePort):
    __slots__ = ("_task_id", "_calls")

    def __init__(self, task_id: TaskId) -> None:
        self._task_id = task_id
        self._calls = 0

    def __bool__(self) -> bool:
        return False

    def resolve_current_task_scope(
        self,
        *,
        run_id: RunId,
        attempt_id: AttemptId,
        execution_id: ExecutionId,
    ) -> TaskId:
        del run_id, attempt_id, execution_id
        self._calls += 1
        return self._task_id


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


def _minimal_side_effect(task_id: TaskId) -> MeaningfulSideEffectRequest:
    return MeaningfulSideEffectRequest(
        action="op",
        kinds=(MeaningfulSideEffectKind.MUTATION,),
        side_effect_scope_id="scope",
        task_id=task_id,
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        principal_id="principal",
        tenant_id="tenant",
        resource="resource",
    )


def test_falsey_custom_task_scope_is_not_replaced_by_default() -> None:
    gate = _minimal_gate()
    scope = _FalseyTaskScope(_TASK)
    boundary = build_default_wired_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        task_scope=scope,
    )
    inner_guard = boundary._inner_execution_guard  # noqa: SLF001 — composition proof
    assert isinstance(inner_guard, DefaultCanonicalInnerExecutionGuard)
    assert inner_guard._task_scope is scope  # noqa: SLF001
    assert not isinstance(inner_guard._task_scope, ActiveTaskRegistryTaskScopeResolver)  # noqa: SLF001


def test_falsey_custom_task_scope_resolve_is_invoked_through_wired_boundary() -> None:
    gate = _minimal_gate()
    scope = _FalseyTaskScope(_TASK)
    boundary = build_default_wired_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        task_scope=scope,
    )
    side_effect = _minimal_side_effect(_TASK)
    request = CollaborativeWorkEnforcementRequest(
        tenant_id="tenant",
        workspace_id="ws",
        operation_id="op",
        acting_principal_id="principal",
        resource_scope="resource",
        membership=None,
        meaningful_side_effect_request=side_effect,
    )
    with bound_gr3_active_execution(
        run_id=side_effect.run_id,
        attempt_id=side_effect.attempt_id,
        execution_id=side_effect.execution_id,
    ):
        boundary.authorize(request)
    assert scope._calls == 1


def test_task_scope_none_selects_active_task_registry_resolver() -> None:
    gate = _minimal_gate()
    boundary = build_default_wired_meaningful_side_effect_authorization_boundary(
        enforcement_gate=gate,
        task_scope=None,
    )
    inner_guard = boundary._inner_execution_guard  # noqa: SLF001
    assert isinstance(inner_guard, DefaultCanonicalInnerExecutionGuard)
    assert isinstance(inner_guard._task_scope, ActiveTaskRegistryTaskScopeResolver)  # noqa: SLF001
