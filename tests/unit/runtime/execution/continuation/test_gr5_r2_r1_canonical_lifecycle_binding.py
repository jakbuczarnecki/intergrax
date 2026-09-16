# © Artur Czarnecki. All rights reserved.

"""GR-5-R2-R1 — query purity, explicit lifecycle driver, canonical progress binding."""

from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    ExecutionContinuationLookup,
    ExecutionContinuationResolutionCommand,
    ExecutionContinuationTransition,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
    advance_continuation_lifecycle,
    execution_continuation_lifecycle_blocks_execution_progress,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.composition import (
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.persistence import (
    InMemoryExecutionContinuationStateStore,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    assert_canonical_execution_may_progress,
    execute_canonical_work_when_unblocked,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SERVICE_MODULE = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "continuation" / "service.py"

_SEED = "gr5-r2-r1"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONTINUATION_ID = "gcr_gr5_r2_r1"


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _pause_request() -> ExecutionPauseRequest:
    return ExecutionPauseRequest(
        identity=_identity(),
        continuation_id=_CONTINUATION_ID,
        reason=ContinuationReason.SECURITY,
    )


def _resolution(expected_revision: int) -> ExecutionContinuationResolutionCommand:
    from intergrax.contracts.human_approver import local_development_approver_evidence

    return ExecutionContinuationResolutionCommand(
        continuation_id=_CONTINUATION_ID,
        identity=_identity(),
        expected_revision=expected_revision,
        verdict=ExecutionHumanVerdict.APPROVE,
        approver=local_development_approver_evidence(
            actor_id="op-r2-r1",
            tenant_id="tenant-gr5-r2-r1",
        ),
        human_request_id="hr_r2_r1",
        resolved_at="2026-09-15T12:00:00Z",
    )


class _InstrumentedStore(InMemoryExecutionContinuationStateStore):
    def __init__(self) -> None:
        super().__init__()
        self.cas_calls = 0
        self.insert_calls = 0

    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        self.insert_calls += 1
        return super().begin_current_episode_if_predecessor_allows(pending)

    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        return super().insert_if_absent(pending)

    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        self.cas_calls += 1
        return super().compare_and_swap(
            continuation_id=continuation_id,
            expected=expected,
            updated=updated,
        )


def _ensure_pause_requested(service: ExecutionContinuationService) -> PendingExecutionContinuation:
    existing = service.store.load(_CONTINUATION_ID)
    if existing is not None:
        return existing
    return service.request_pause(_pause_request())


def _set_state(
    service: ExecutionContinuationService,
    state: ExecutionContinuationLifecycleState,
) -> PendingExecutionContinuation:
    pending = _ensure_pause_requested(service)
    driver = ExecutionContinuationLifecycleDriver(service)
    if state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED:
        return pending
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
        from intergrax.contracts.execution_continuation import ExecutionContinuationResumeCommand

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


@pytest.mark.parametrize(
    "state",
    [
        ExecutionContinuationLifecycleState.PAUSE_REQUESTED,
        ExecutionContinuationLifecycleState.PAUSED,
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
    ],
)
def test_get_pending_is_pure_read(state: ExecutionContinuationLifecycleState) -> None:
    store = _InstrumentedStore()
    service = ExecutionContinuationService(store)
    _set_state(service, state)
    before = store.load(_CONTINUATION_ID)
    assert before is not None
    cas_before = store.cas_calls
    insert_before = store.insert_calls
    lookup = ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID)
    for _ in range(5):
        snapshot = service.get_pending(lookup)
        assert snapshot.model_dump() == before.model_dump()
    after = store.load(_CONTINUATION_ID)
    assert after == before
    assert store.cas_calls == cas_before
    assert store.insert_calls == insert_before


def test_get_pending_performs_no_cas_on_instrumented_store() -> None:
    store = _InstrumentedStore()
    service = ExecutionContinuationService(store)
    service.request_pause(_pause_request())
    store.cas_calls = 0
    store.insert_calls = 0
    service.get_pending(ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID))
    assert store.cas_calls == 0
    assert store.insert_calls == 0


def test_request_pause_does_not_auto_advance_to_waiting() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    pending = deps.continuation.request_pause(_pause_request())
    assert pending.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED
    loaded = deps.continuation.get_pending(
        ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID),
    )
    assert loaded.lifecycle_state is ExecutionContinuationLifecycleState.PAUSE_REQUESTED


def test_pause_fact_requires_execution_pause_established() -> None:
    service = ExecutionContinuationService(InMemoryExecutionContinuationStateStore())
    service.request_pause(_pause_request())
    driver = ExecutionContinuationLifecycleDriver(service)
    with pytest.raises(ExecutionContinuationError) as exc:
        driver.record_execution_reached_safe_pause(
            _CONTINUATION_ID,
            execution_pause_established=False,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION


def test_apply_resolution_from_pause_requested_fails() -> None:
    service = ExecutionContinuationService(InMemoryExecutionContinuationStateStore())
    service.request_pause(_pause_request())
    with pytest.raises(ExecutionContinuationError) as exc:
        service.apply_resolution(_resolution(expected_revision=1))
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION


def test_apply_resolution_from_paused_fails_without_waiting() -> None:
    service = ExecutionContinuationService(InMemoryExecutionContinuationStateStore())
    driver = ExecutionContinuationLifecycleDriver(service)
    service.request_pause(_pause_request())
    paused = driver.record_execution_reached_safe_pause(
        _CONTINUATION_ID,
        execution_pause_established=True,
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        service.apply_resolution(_resolution(expected_revision=paused.revision))
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION


@pytest.mark.parametrize(
    "state,blocks",
    [
        (ExecutionContinuationLifecycleState.PAUSE_REQUESTED, False),
        (ExecutionContinuationLifecycleState.PAUSED, True),
        (ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN, True),
        (ExecutionContinuationLifecycleState.RESUME_AUTHORIZED, True),
        (ExecutionContinuationLifecycleState.RESUMED, False),
    ],
)
def test_progress_gate_semantics(
    state: ExecutionContinuationLifecycleState,
    blocks: bool,
) -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, state)
    if blocks:
        with pytest.raises(ExecutionContinuationError) as exc:
            assert_canonical_execution_may_progress(store=store, identity=_identity())
        assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED
    else:
        assert_canonical_execution_may_progress(store=store, identity=_identity())


@pytest.mark.asyncio
async def test_waiting_blocks_canonical_work_execution() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
    ran = {"count": 0}

    async def _work() -> str:
        ran["count"] += 1
        return "done"

    with pytest.raises(ExecutionContinuationError):
        await execute_canonical_work_when_unblocked(
            store=store,
            identity=_identity(),
            work=_work,
        )
    assert ran["count"] == 0


@pytest.mark.asyncio
async def test_resumed_allows_canonical_work_execution() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.RESUMED)

    async def _work() -> str:
        return "ok"

    result = await execute_canonical_work_when_unblocked(
        store=store,
        identity=_identity(),
        work=_work,
    )
    assert result == "ok"


def test_concurrent_get_pending_does_not_mutate() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _set_state(service, ExecutionContinuationLifecycleState.PAUSED)
    before = store.load(_CONTINUATION_ID)
    assert before is not None
    lookup = ExecutionContinuationLookup(continuation_id=_CONTINUATION_ID)
    barrier = threading.Barrier(4)

    def _read() -> None:
        barrier.wait()
        service.get_pending(lookup)

    threads = [threading.Thread(target=_read) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    after = store.load(_CONTINUATION_ID)
    assert after == before


def test_get_pending_does_not_call_lifecycle_commit_helpers() -> None:
    tree = ast.parse(_SERVICE_MODULE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_pending":
            source = ast.get_source_segment(_SERVICE_MODULE.read_text(encoding="utf-8"), node) or ""
            assert "_commit_lifecycle_transition" not in source
            assert "_advance_one_internal_step" not in source
            assert "_ensure_waiting_for_human" not in source
            return
    raise AssertionError("get_pending not found")


def test_contract_blocks_progress_helper_matches_gate() -> None:
    for state in ExecutionContinuationLifecycleState:
        expected = state in {
            ExecutionContinuationLifecycleState.PAUSED,
            ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            ExecutionContinuationLifecycleState.RESUME_AUTHORIZED,
            ExecutionContinuationLifecycleState.REJECTED,
            ExecutionContinuationLifecycleState.ESCALATED,
            ExecutionContinuationLifecycleState.CANCELLED,
        }
        assert execution_continuation_lifecycle_blocks_execution_progress(state) is expected


def test_four_ids_preserved_through_driver_path() -> None:
    deps = wire_execution_engine_continuation_dependencies()
    waiting = _set_state(deps.continuation_service, ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN)
    assert waiting.identity.task_id == _TASK
    assert waiting.identity.run_id == _RUN
    assert waiting.identity.attempt_id == _ATTEMPT
    assert waiting.identity.execution_id == _EXECUTION
