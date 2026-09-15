# © Artur Czarnecki. All rights reserved.

"""GR-3-R2 — explicit ActiveExecutionTaskScopePort composition (no guard fallback)."""

from __future__ import annotations

import inspect
import pytest

from intergrax.contracts.active_execution_task_scope import ActiveExecutionTaskScopePort
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
    build_canonical_inner_execution_guard,
    build_default_canonical_inner_execution_guard,
)
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistryTaskScopeResolver
from tests.unit.runtime.governance.gr3_test_support import bound_gr3_active_execution

pytestmark = pytest.mark.unit

_TASK_A = mint_task_id()
_TASK_B = mint_task_id()


class _StaticTaskScope(ActiveExecutionTaskScopePort):
    __slots__ = ("_task_id", "_calls")

    def __init__(self, task_id: TaskId) -> None:
        self._task_id = task_id
        self._calls = 0

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


def _minimal_request(task_id: TaskId) -> MeaningfulSideEffectRequest:
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


def test_guard_constructor_requires_task_scope() -> None:
    params = inspect.signature(DefaultCanonicalInnerExecutionGuard.__init__).parameters
    task_scope_param = params["task_scope"]
    assert task_scope_param.default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        DefaultCanonicalInnerExecutionGuard()  # type: ignore[call-arg]


def test_custom_task_scope_is_invoked_by_guard() -> None:
    scope = _StaticTaskScope(_TASK_A)
    guard = DefaultCanonicalInnerExecutionGuard(task_scope=scope)
    request = _minimal_request(_TASK_A)
    with bound_gr3_active_execution(
        run_id=request.run_id,
        attempt_id=request.attempt_id,
        execution_id=request.execution_id,
    ):
        guard.assert_meaningful_side_effect_bound(request)
    assert scope._calls == 1


def test_two_task_scope_implementations_same_guard_class() -> None:
    scope_a = _StaticTaskScope(_TASK_A)
    scope_b = _StaticTaskScope(_TASK_B)
    guard_a = DefaultCanonicalInnerExecutionGuard(task_scope=scope_a)
    guard_b = DefaultCanonicalInnerExecutionGuard(task_scope=scope_b)
    assert type(guard_a) is type(guard_b)
    assert guard_a.__class__ is guard_b.__class__


def test_default_builder_uses_active_task_registry_resolver() -> None:
    guard = build_default_canonical_inner_execution_guard()
    assert isinstance(guard, DefaultCanonicalInnerExecutionGuard)
    inner_scope = guard._task_scope  # noqa: SLF001 — composition proof
    assert isinstance(inner_scope, ActiveTaskRegistryTaskScopeResolver)


def test_build_canonical_inner_execution_guard_accepts_custom_scope() -> None:
    scope = _StaticTaskScope(_TASK_A)
    guard = build_canonical_inner_execution_guard(task_scope=scope)
    assert guard._task_scope is scope  # noqa: SLF001
