# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R2 — mandatory continuation gate on canonical ExecutionBoundary spine."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from typing import Any

import pytest

from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationResumeCommand,
    ExecutionContinuationTransition,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    advance_continuation_lifecycle,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BOUNDARY_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "boundary.py"

_SEED = "gr5-r2-r2"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_OTHER_ATTEMPT = mint_attempt_id()
_OTHER_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r2_r2"
_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


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


def _pause_request() -> ExecutionPauseRequest:
    return ExecutionPauseRequest(
        identity=_identity(),
        continuation_id=_CONTINUATION_ID,
        reason=ContinuationReason.SECURITY,
    )


def _resolution(expected_revision: int) -> ExecutionContinuationResolutionCommand:
    return ExecutionContinuationResolutionCommand(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        expected_revision=expected_revision,
        verdict=ExecutionHumanVerdict.APPROVE,
        approver=local_development_approver_evidence(
            actor_id="op-r2-r2",
            tenant_id="tenant-gr5-r2-r2",
        ),
        human_request_id="hr_r2_r2",
        resolved_at="2026-09-15T12:00:00Z",
    )


def _set_state(
    service: ExecutionContinuationService,
    state: ExecutionContinuationLifecycleState,
) -> PendingExecutionContinuation:
    existing = service.store.load(_CONTINUATION_ID)
    if existing is None:
        service.request_pause(_pause_request())
    driver = ExecutionContinuationLifecycleDriver(service)
    if state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
        return service.store.load(_CONTINUATION_ID)  # type: ignore[return-value]
    if state is ExecutionContinuationLifecycleState.PAUSED:
        return driver.record_execution_reached_safe_pause(
            _CONTINUATION_ID,
            execution_pause_established=True,
        )
    if state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN:
        driver.record_execution_reached_safe_pause(
            _CONTINUATION_ID,
            execution_pause_established=True,
        )
        return driver.record_ready_for_human_resolution(_CONTINUATION_ID)
    if state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED:
        waiting = _set_state(service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
        return service.apply_resolution(_resolution(waiting.revision))
    if state is ExecutionContinuationLifecycleState.RESUMED:
        authorized = _set_state(service, ExecutionContinuationLifecycleState.RESUME_AUTHORIZED)
        return service.resume(
            ExecutionContinuationResumeCommand(
                continuation_id=_CONTINUATION_ID,
                identity=_identity(),
                expected_revision=authorized.revision,
            ),
        )
    waiting = _set_state(service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
    terminal = advance_continuation_lifecycle(
        waiting.lifecycle_state,
        ExecutionContinuationTransition.CANCEL_CONTINUATION,
    )
    updated = waiting.model_copy(
        update={"lifecycle_state": terminal, "revision": waiting.revision + 1},
    )
    assert service.store.compare_and_swap(
        continuation_id=_CONTINUATION_ID,
        expected=waiting,
        updated=updated,
    )
    return updated


class _CountingDelegate:
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count = 0

    async def execute(self, request: object) -> str:
        self.count += 1
        return f"ok:{request!r}"


def _root_context() -> RootExecutionContext:
    return RootExecutionContext(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        authority=_AUTHORITY,
        task_id=_TASK,
    )


def _binding() -> ExecutionIdentityBinding:
    return ExecutionIdentityBinding(
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
        task_id=_TASK,
    )


async def _run_through_runtime(
    store: ExecutionContinuationStateStore,
    delegate: _CountingDelegate,
) -> None:
    runtime = ExecutionRuntime(delegate, continuation_state_store=store)
    await runtime.execute("probe", _root_context())


async def _run_through_boundary(
    store: ExecutionContinuationStateStore,
    delegate: _CountingDelegate,
) -> None:
    boundary = ExecutionBoundary(
        delegate,
        identity=_binding(),
        authority=_AUTHORITY,
        continuation_state_store=store,
    )
    await boundary.execute("probe")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state",
    [
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
        ExecutionContinuationLifecycleState.REJECTED,
        ExecutionContinuationLifecycleState.ESCALATED,
        ExecutionContinuationLifecycleState.CANCELLED,
    ],
)
async def test_runtime_boundary_blocks_all_blocking_states(
    state: ExecutionContinuationLifecycleState,
) -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, state)
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await _run_through_runtime(store, delegate)
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED
    assert delegate.count == 0


@pytest.mark.asyncio
async def test_runtime_boundary_allows_resumed_work_once() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.RESUMED)
    delegate = _CountingDelegate()
    result = await ExecutionRuntime(delegate, continuation_state_store=store).execute(
        "probe",
        _root_context(),
    )
    assert delegate.count == 1
    assert result.startswith("ok:")


@pytest.mark.asyncio
async def test_pause_requested_allows_one_unit_then_advances_to_paused() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.PAUSE_REQUESTED)
    delegate = _CountingDelegate()
    await _run_through_runtime(store, delegate)
    assert delegate.count == 1
    pending = service.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSED


@pytest.mark.asyncio
async def test_no_continuation_allows_progress() -> None:
    store = InMemoryExecutionContinuationStateStore()
    delegate = _CountingDelegate()
    await _run_through_runtime(store, delegate)
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_wrong_execution_identity_does_not_block() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    other_identity = _identity(execution_id=_OTHER_EXECUTION)
    service.request_pause(
        ExecutionPauseRequest(
            identity=other_identity,
            continuation_id="gcr_other_exec",
            reason=ContinuationReason.SECURITY,
        ),
    )
    driver = ExecutionContinuationLifecycleDriver(service)
    driver.record_execution_reached_safe_pause("gcr_other_exec", execution_pause_established=True)
    driver.record_ready_for_human_resolution("gcr_other_exec")
    delegate = _CountingDelegate()
    await _run_through_runtime(store, delegate)
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_different_attempt_does_not_block() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    other_attempt_id = _OTHER_ATTEMPT
    service.request_pause(
        ExecutionPauseRequest(
            identity=_identity(attempt_id=other_attempt_id),
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
    await _run_through_runtime(store, delegate)
    assert delegate.count == 1


@pytest.mark.asyncio
async def test_ambiguous_identity_fails_closed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(_pause_request())
    duplicate = _pause_request().model_copy(update={"continuation_id": "gcr_dup"})
    service.store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id="gcr_dup",
            identity=_identity(),
            lifecycle_state=ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
            revision=1,
            reason=ContinuationReason.SECURITY,
        ),
    )
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await _run_through_runtime(store, delegate)
    assert exc.value.code is ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY
    assert delegate.count == 0


class _FailingStore(InMemoryExecutionContinuationStateStore):
    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        raise RuntimeError("store unavailable")


@pytest.mark.asyncio
async def test_store_query_failure_fails_closed() -> None:
    store = _FailingStore()
    delegate = _CountingDelegate()
    with pytest.raises(ExecutionContinuationError) as exc:
        await _run_through_boundary(store, delegate)
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    assert delegate.count == 0


def test_safe_pause_provenance_only_from_boundary_quiescence() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(_pause_request())
    driver = ExecutionContinuationLifecycleDriver(service)
    with pytest.raises(ExecutionContinuationError):
        driver.record_execution_reached_safe_pause(
            _CONTINUATION_ID,
            execution_pause_established=False,
        )


def test_boundary_source_invokes_progress_gate() -> None:
    source = _BOUNDARY_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "assert_canonical_execution_may_progress"
    ]
    assert calls, "ExecutionBoundary must call assert_canonical_execution_may_progress"
    assert "ExecutionContinuationLifecycleDriver" in source
    assert "NexusPort" not in source


def test_boundary_module_has_no_public_nexus_contract_imports() -> None:
    tree = ast.parse(_BOUNDARY_MODULE.read_text(encoding="utf-8"))
    forbidden = (
        "NexusPort",
        "NexusExecutionPort",
        "NexusProgressPort",
        "NexusContinuationPort",
        "PublicNexusFacade",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "nexus" in node.module:
            pytest.fail(f"unexpected Nexus import in ExecutionBoundary: {node.module}")
        if isinstance(node, ast.Name) and node.id in forbidden:
            pytest.fail(f"forbidden Nexus symbol in ExecutionBoundary: {node.id}")


def test_get_pending_purity_regression() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
    before = store.load(_CONTINUATION_ID)
    for _ in range(3):
        assert service.get_pending(
            ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID),
        ).model_dump() == before.model_dump()
    assert store.load(_CONTINUATION_ID) == before


def test_concurrent_resume_max_one_success_regression() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    waiting = _set_state(service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
    approved = service.apply_resolution(_resolution(waiting.revision))
    barrier = threading.Barrier(2)
    results: list[Any] = []
    command = ExecutionContinuationResumeCommand(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        expected_revision=approved.revision,
    )

    def _resume() -> None:
        barrier.wait()
        try:
            results.append(service.resume(command))
        except ExecutionContinuationError as exc:
            results.append(exc)

    threads = [threading.Thread(target=_resume), threading.Thread(target=_resume)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    successes = [item for item in results if isinstance(item, PendingExecutionContinuation)]
    stale = [
        item
        for item in results
        if isinstance(item, ExecutionContinuationError)
        and item.code is ExecutionContinuationErrorCode.STALE_REVISION
    ]
    assert len(successes) == 1
    assert len(stale) == 1
