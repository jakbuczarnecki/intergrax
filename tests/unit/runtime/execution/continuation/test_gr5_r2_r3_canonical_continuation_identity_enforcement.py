# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R3 — authoritative continuation identity without Task registry fallback."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionPauseRequest,
)
from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BOUNDARY = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"
_CHILD = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "child.py"

_SEED = "gr5-r2-r3"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_OTHER_EXECUTION = mint_execution_id()
_OTHER_ATTEMPT = mint_attempt_id()
_CONTINUATION_ID = "gcr_gr5_r2_r3"
_AUTHORITY = ParentExecutionAuthority.unrestricted_root()
_LEDGER = create_execution_budget_ledger(RunBudget())


def _identity(
    *,
    task_id: str | None = None,
    attempt_id: str | None = None,
    execution_id: str | None = None,
) -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=task_id or _TASK,
        run_id=_RUN,
        attempt_id=attempt_id or _ATTEMPT,
        execution_id=execution_id or _EXECUTION,
    )


def _binding(*, task_id: str | None = _TASK) -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        task_id=task_id,
    )


def _root_context(*, task_id: str | None = _TASK) -> RootExecutionContext:
    return RootExecutionContext(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        authority=_AUTHORITY,
        task_id=task_id,
    )


class _CountingDelegate:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    async def execute(self, request: object) -> str:
        self.count += 1
        return "ok"


def _block_execution(store: InMemoryExecutionContinuationStateStore) -> None:
    service = ExecutionContinuationService(store)
    service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
        ),
    )
    driver = ExecutionContinuationLifecycleDriver(service)
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    driver.record_ready_for_human_resolution(_CONTINUATION_ID)


def _module_imports_task_runtime(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "intergrax.runtime.task" or node.module.startswith(
                "intergrax.runtime.task.",
            ):
                hits.append(node.module)
    return hits


@pytest.mark.asyncio
async def test_root_full_identity_evaluates_exact_continuation() -> None:
    store = InMemoryExecutionContinuationStateStore()
    _block_execution(store)
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await ExecutionRuntime(delegate, continuation_state_store=store).execute(
            "probe",
            _root_context(),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED
    assert delegate.count == 0


@pytest.mark.asyncio
async def test_missing_root_task_id_fails_closed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await ExecutionRuntime(delegate, continuation_state_store=store).execute(
            "probe",
            _root_context(task_id=None),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.INCOMPLETE_EXECUTION_IDENTITY
    assert delegate.count == 0


@pytest.mark.asyncio
async def test_no_registry_fallback_when_binding_lacks_task_id() -> None:
    store = InMemoryExecutionContinuationStateStore()
    registry_task = mint_task_id()
    task = Task(
        task_id=registry_task,
        tenant_id="t",
        user_id="u",
        message="registry decoy",
        context=TaskContext(),
    )
    await ActiveTaskRegistry.register(task, _RUN)
    delegate = _CountingDelegate()
    boundary = ExecutionBoundary(
        delegate,
        identity=_binding(task_id=None),
        authority=_AUTHORITY,
        continuation_state_store=store,
    )
    try:
        with pytest.raises(ExecutionContinuationError) as exc:
            await boundary.execute("probe")
        assert exc.value.code is ExecutionContinuationErrorCode.INCOMPLETE_EXECUTION_IDENTITY
        assert delegate.count == 0
    finally:
        await ActiveTaskRegistry.unregister(registry_task, _RUN)


@pytest.mark.asyncio
async def test_empty_registry_root_enforcement_works() -> None:
    store = InMemoryExecutionContinuationStateStore()
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1


@dataclass(frozen=True)
class _Ping:
    value: str


@dataclass(frozen=True)
class _Pong:
    value: str


@pytest.mark.asyncio
async def test_child_task_id_propagation_without_registry() -> None:
    child_bindings: list[ExecutionIdentityBinding] = []
    real_init = ExecutionBoundary.__init__

    def _capture_init(self: ExecutionBoundary[Any, Any], *args: Any, **kwargs: Any) -> None:
        identity = kwargs.get("identity")
        if identity is not None and identity.parent_execution_id is not None:
            child_bindings.append(identity)
        real_init(self, *args, **kwargs)

    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        task_id=_TASK,
    )
    child_runner = ChildExecutionRunner[_Ping, _Pong](ledger=_LEDGER)

    class _RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            return await child_runner.execute(
                request=request,
                delegate=_ChildEcho(),
            )

    with patch.object(ExecutionBoundary, "__init__", _capture_init):
        await ExecutionBoundary[_Ping, _Pong](
            _RootDelegate(),
            identity=root,
            authority=_AUTHORITY,
        ).execute(_Ping(value="child"))

    assert len(child_bindings) == 1
    assert child_bindings[0].task_id == _TASK
    assert child_bindings[0].run_id == root.run_id
    assert child_bindings[0].attempt_id == root.attempt_id


class _ChildEcho:
    async def execute(self, request: _Ping) -> _Pong:
        return _Pong(value=request.value)


@pytest.mark.asyncio
async def test_active_context_task_id_nested_scope_restored() -> None:
    root = _binding()
    token = bind_active_execution_identity(
        run_id=root.run_id,
        attempt_id=root.attempt_id,
        execution_id=root.execution_id,
        task_id=root.task_id,
    )
    assert peek_active_execution_task_id() == _TASK
    reset_active_execution_identity(token)
    assert peek_active_execution_task_id() is None


@pytest.mark.asyncio
async def test_same_task_different_execution_does_not_block_child() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(execution_id=_OTHER_EXECUTION),
            continuation_id="gcr_other_exec",
            reason=ContinuationReason.SECURITY,
        ),
    )
    driver = ExecutionContinuationLifecycleDriver(service)
    driver.record_execution_reached_safe_pause("gcr_other_exec", execution_pause_established=True)
    driver.record_ready_for_human_resolution("gcr_other_exec")
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_different_attempt_isolation() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(attempt_id=_OTHER_ATTEMPT),
            continuation_id="gcr_other_attempt",
            reason=ContinuationReason.SECURITY,
        ),
    )
    driver = ExecutionContinuationLifecycleDriver(service)
    driver.record_execution_reached_safe_pause(
        "gcr_other_attempt",
        execution_pause_established=True,
    )
    driver.record_ready_for_human_resolution("gcr_other_attempt")
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_boundary_without_store_allows_missing_task_id() -> None:
    delegate = _CountingDelegate()
    boundary = ExecutionBoundary(
        delegate,
        identity=_binding(task_id=None),
        authority=_AUTHORITY,
        continuation_state_store=None,
    )
    await boundary.execute("probe")
    assert delegate.count == 1


def test_boundary_module_has_no_task_runtime_import_for_continuation() -> None:
    assert _module_imports_task_runtime(_BOUNDARY) == []


def test_child_module_has_no_active_task_registry_import() -> None:
    source = _CHILD.read_text(encoding="utf-8")
    assert "active_task_registry" not in source
    assert _module_imports_task_runtime(_CHILD) == []


def test_get_pending_purity_regression() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(),
            continuation_id=_CONTINUATION_ID,
            reason=ContinuationReason.SECURITY,
        ),
    )
    driver = ExecutionContinuationLifecycleDriver(service)
    driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    waiting = driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    before = store.load(_CONTINUATION_ID)
    for _ in range(3):
        assert service.get_pending(
            ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID),
        ).model_dump() == before.model_dump()
    assert waiting.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
