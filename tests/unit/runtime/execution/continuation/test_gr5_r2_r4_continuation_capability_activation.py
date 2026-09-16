# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R4 — explicit continuation capability activation vs identity compatibility."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationResolutionCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_continuation_state_store import (
    ExecutionContinuationStateStore,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.runtime.execution.active_execution_continuation_store import (
    bind_active_execution_continuation_state_store,
    peek_active_execution_continuation_state_store as peek_active_store,
    reset_active_execution_continuation_state_store,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
    wire_execution_continuation_state_store,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_RUNTIME = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "runtime.py"
_NEXUS_HOST = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "nexus_host_execution.py"

_SEED = "gr5-r2-r4"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r2_r4"
_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


def _root_context(*, task_id: str | None = _TASK) -> RootExecutionContext:
    return RootExecutionContext(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        authority=_AUTHORITY,
        task_id=task_id,
    )


def _binding(*, task_id: str | None = _TASK) -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        task_id=task_id,
    )


class _CountingDelegate:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    async def execute(self, request: object) -> str:
        self.count += 1
        return "ok"


class _FalseyStore(ExecutionContinuationStateStore):
    @property
    def is_durable(self) -> bool:
        return False

    def __bool__(self) -> bool:
        return False

    def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
        return None

    def find_by_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return None

    def resolve_current_episode_for_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return None

    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return None

    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        return True

    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        return True

    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        return False


def _block_execution(store: InMemoryExecutionContinuationStateStore) -> None:
    service = ExecutionContinuationService(store)
    service.request_pause(
        ExecutionPauseRequest(
            identity=ExecutionContinuationIdentity(
                task_id=_TASK,
                run_id=_RUN,
                attempt_id=_ATTEMPT,
                execution_id=_EXECUTION,
            ),
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


@pytest.mark.asyncio
async def test_disabled_root_without_task_id_executes() -> None:
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate).execute("probe", _root_context(task_id=None))
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_enabled_root_without_task_id_fails_closed() -> None:
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
async def test_enabled_full_identity_empty_store_executes() -> None:
    store = InMemoryExecutionContinuationStateStore()
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_enabled_blocked_by_waiting_for_human() -> None:
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
async def test_enabled_resumed_executes_once() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    identity = ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )
    service.request_pause(
        ExecutionPauseRequest(
            identity=identity,
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
    authorized = service.apply_resolution(
        ExecutionContinuationResolutionCommand(
            continuation_id=_CONTINUATION_ID,
            identity=identity,
            expected_revision=waiting.revision,
            verdict=ExecutionHumanVerdict.APPROVE,
            approver=local_development_approver_evidence(
                actor_id="op-r2-r4",
                tenant_id="tenant-gr5-r2-r4",
            ),
            human_request_id="hr_r2_r4",
            resolved_at="2026-09-15T12:00:00Z",
        ),
    )
    service.resume(
        ExecutionContinuationResumeCommand(
            continuation_id=_CONTINUATION_ID,
            identity=identity,
            expected_revision=authorized.revision,
        ),
    )
    delegate = _CountingDelegate()
    await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_disabled_does_not_query_continuation_store() -> None:
    delegate = _CountingDelegate()
    with patch.object(
        InMemoryExecutionContinuationStateStore,
        "resolve_identity_for_execution_progress",
        side_effect=AssertionError("must not query"),
    ):
        await ExecutionRuntime(delegate).execute("probe", _root_context(task_id=None))
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_explicit_falsey_store_enables_capability() -> None:
    falsey = _FalseyStore()
    runtime = ExecutionRuntime(_CountingDelegate(), continuation_state_store=falsey)
    assert runtime._continuation_state_store is falsey
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await ExecutionRuntime(delegate, continuation_state_store=falsey).execute(
            "probe",
            _root_context(task_id=None),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.INCOMPLETE_EXECUTION_IDENTITY
    assert delegate.count == 0


def test_default_general_runtime_has_no_implicit_store() -> None:
    runtime = ExecutionRuntime(_CountingDelegate())
    assert runtime._continuation_state_store is None


def test_dedicated_continuation_composition_provides_default_store() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    assert isinstance(
        deps.continuation_service.store,
        InMemoryExecutionContinuationStateStore,
    )


def test_wire_helper_with_falsey_store_preserves_enabled() -> None:
    falsey = _FalseyStore()
    deps = wire_execution_engine_continuation_dependencies(state_store=falsey)
    assert deps.continuation_service.store is falsey


@pytest.mark.asyncio
async def test_active_store_reset_after_enabled_then_disabled() -> None:
    store = InMemoryExecutionContinuationStateStore()
    delegate_a = _CountingDelegate()
    await ExecutionRuntime(delegate_a, continuation_state_store=store).execute(
        "a",
        _root_context(),
    )
    assert peek_active_store() is None
    delegate_b = _CountingDelegate()
    await ExecutionRuntime(delegate_b).execute("b", _root_context(task_id=None))
    assert delegate_b.count == 1
    assert peek_active_store() is None


@dataclass(frozen=True)
class _Ping:
    value: str


@dataclass(frozen=True)
class _Pong:
    value: str


@pytest.mark.asyncio
async def test_nested_child_inherits_enabled_store() -> None:
    store = InMemoryExecutionContinuationStateStore()
    ledger = create_execution_budget_ledger(RunBudget())
    child_runner = ChildExecutionRunner[_Ping, _Pong](
        ledger=ledger,
        continuation_state_store=store,
    )
    root = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        task_id=_TASK,
    )

    class _RootDelegate:
        async def execute(self, request: _Ping) -> _Pong:
            return await child_runner.execute(request=request, delegate=_ChildEcho())

    class _ChildEcho:
        async def execute(self, request: _Ping) -> _Pong:
            return _Pong(value=request.value)

    token = bind_active_execution_continuation_state_store(store)
    try:
        await ExecutionBoundary[_Ping, _Pong](
            _RootDelegate(),
            identity=root,
            authority=_AUTHORITY,
            continuation_state_store=store,
        ).execute(_Ping(value="nested"))
    finally:
        reset_active_execution_continuation_state_store(token)


def test_execution_runtime_does_not_wire_implicit_default_store() -> None:
    init_source = _RUNTIME.read_text(encoding="utf-8").split("def __init__", 1)[1]
    assert "wire_execution_continuation_state_store" not in init_source


def test_nexus_host_execution_explicitly_enables_continuation() -> None:
    source = _NEXUS_HOST.read_text(encoding="utf-8")
    assert "wire_execution_continuation_state_store" in source
    assert "_continuation_state_store=" in source


def test_build_host_task_execution_type_has_continuation_field() -> None:
    tree = ast.parse(_NEXUS_HOST.read_text(encoding="utf-8"))
    host_field_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "HostTaskExecution":
                for kw in node.keywords:
                    if kw.arg == "_continuation_state_store":
                        host_field_found = True
    assert host_field_found


@pytest.mark.asyncio
async def test_boundary_explicit_none_store_skips_gate() -> None:
    delegate = _CountingDelegate()
    boundary = ExecutionBoundary(
        delegate,
        identity=_binding(task_id=None),
        authority=_AUTHORITY,
        continuation_state_store=None,
    )
    with patch(
        "intergrax.runtime.execution.boundary.assert_canonical_execution_may_progress",
    ) as gate:
        await boundary.execute("probe")
        gate.assert_not_called()
    assert delegate.count == 1
