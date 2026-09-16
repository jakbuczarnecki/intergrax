# © Artur Czarnecki. All rights reserved.

"""GR-5-R5 — durable process restart + exact execution identity qualification."""

from __future__ import annotations

import ast
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    execution_continuation_recovery_handle_for_continuation_id,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.continuation.composition import (
    reconnect_execution_engine_continuation_dependencies,
    wire_execution_engine_continuation_dependencies,
)
from intergrax.runtime.execution.continuation.lifecycle_driver import (
    ExecutionContinuationLifecycleDriver,
)
from intergrax.runtime.execution.continuation.persistence import (
    ExecutionContinuationDurableBacking,
    InMemoryExecutionContinuationStateStore,
    backing_execution_continuation_state_store,
    execution_continuation_state_store_from_durable_export,
    export_durable_continuation_state,
    restore_durable_continuation_backing,
)
from intergrax.runtime.execution.continuation.progress_gate import (
    assert_canonical_execution_may_progress,
)
from intergrax.runtime.execution.continuation.restart_qualification import (
    qualify_execution_continuation_process_restart,
    recover_execution_continuation_process_restart,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.execution.runtime import ExecutionRuntime
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.task.execution_continuation_projection import (
    execution_continuation_projection_payload_digest,
    wire_task_execution_continuation_projection_sink,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SEED = "gr5-r5-restart"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_C1 = "cont_r5_c1"
_C2 = "cont_r5_c2"
_C3 = "cont_r5_c3"
_AUTHORITY = ParentExecutionAuthority.unrestricted_root()
_TENANT = "tenant-gr5-r5"


def _task() -> Task:
    return Task(
        task_id=_TASK,
        tenant_id=_TENANT,
        user_id="user-gr5-r5",
        message="gr5-r5",
        run_id=_RUN,
    )


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _pause_request(continuation_id: str) -> ExecutionPauseRequest:
    return ExecutionPauseRequest(
        identity=_identity(),
        continuation_id=continuation_id,
        reason=ContinuationReason.SECURITY,
        pause_id=f"pause_{continuation_id}",
        human_request_id=f"hr_{continuation_id}",
    )


def _resolution(
    continuation_id: str,
    *,
    expected_revision: int,
    verdict: ExecutionHumanVerdict = ExecutionHumanVerdict.APPROVE,
) -> ExecutionContinuationResolutionCommand:
    return ExecutionContinuationResolutionCommand(
        continuation_id=continuation_id,
        identity=_identity(),
        expected_revision=expected_revision,
        verdict=verdict,
        approver=local_development_approver_evidence(
            actor_id="op-r5",
            tenant_id=_TENANT,
        ),
        human_request_id=f"hr_{continuation_id}",
        pause_id=f"pause_{continuation_id}",
        resolved_at="2026-09-16T12:00:00Z",
    )


def _resume(continuation_id: str, *, expected_revision: int) -> ExecutionContinuationResumeCommand:
    return ExecutionContinuationResumeCommand(
        continuation_id=continuation_id,
        identity=_identity(),
        expected_revision=expected_revision,
    )


@dataclass
class _ProcessA:
    backing: ExecutionContinuationDurableBacking
    store: ExecutionContinuationStateStore
    service: ExecutionContinuationService
    driver: ExecutionContinuationLifecycleDriver


@dataclass
class _ProcessB:
    backing: ExecutionContinuationDurableBacking
    store: ExecutionContinuationStateStore
    service: ExecutionContinuationService
    driver: ExecutionContinuationLifecycleDriver


def _spawn_process_a() -> _ProcessA:
    backing = ExecutionContinuationDurableBacking()
    store = backing_execution_continuation_state_store(backing)
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    return _ProcessA(
        backing=backing,
        store=store,
        service=deps.continuation_service,
        driver=deps.lifecycle_driver,
    )


def _true_restart_process_b(proc_a: _ProcessA) -> _ProcessB:
    payload = export_durable_continuation_state(proc_a.backing)
    backing_a = proc_a.backing
    store_a = proc_a.store
    service_a = proc_a.service
    lock_a = backing_a._lock
    del proc_a, service_a
    backing_b = restore_durable_continuation_backing(payload)
    assert backing_b is not backing_a
    assert backing_b._lock is not lock_a
    store_b = execution_continuation_state_store_from_durable_export(payload)
    assert store_b is not store_a
    assert store_b.is_durable is True
    deps_b = reconnect_execution_engine_continuation_dependencies(state_store=store_b)
    return _ProcessB(
        backing=backing_b,
        store=store_b,
        service=deps_b.continuation_service,
        driver=deps_b.lifecycle_driver,
    )


def _recover(store: ExecutionContinuationStateStore, continuation_id: str):
    return recover_execution_continuation_process_restart(
        store=store,
        recovery_handle=execution_continuation_recovery_handle_for_continuation_id(
            continuation_id,
        ),
        authority=_AUTHORITY,
        tenant_id=_TENANT,
        task_id_consistency=_TASK,
    )


def _drive_to_waiting(
    service: ExecutionContinuationService,
    driver: ExecutionContinuationLifecycleDriver,
    continuation_id: str,
) -> PendingExecutionContinuation:
    service.request_pause(_pause_request(continuation_id))
    driver.record_execution_reached_safe_pause(
        continuation_id,
        execution_pause_established=True,
    )
    return driver.record_ready_for_human_resolution(continuation_id)


def _drive_to_resumed(
    service: ExecutionContinuationService,
    driver: ExecutionContinuationLifecycleDriver,
    continuation_id: str,
) -> PendingExecutionContinuation:
    waiting = _drive_to_waiting(service, driver, continuation_id)
    approved = service.apply_resolution(
        _resolution(continuation_id, expected_revision=waiting.revision),
    )
    return service.resume(_resume(continuation_id, expected_revision=approved.revision))


def test_waiting_restart_current_c2_blocked() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    assert waiting.revision >= 3
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    assert qual.current_episode.continuation_id == _C2
    assert qual.current_episode.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    assert qual.current_episode.revision == waiting.revision
    assert qual.identity == _identity()
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_canonical_execution_may_progress(store=proc_b.store, identity=qual.identity)
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED
    assert proc_b.service is not None


def test_approve_after_restart_same_ids() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    approved = proc_b.service.apply_resolution(
        _resolution(_C2, expected_revision=qual.current_episode.revision),
    )
    resumed = proc_b.service.resume(_resume(_C2, expected_revision=approved.revision))
    assert resumed.lifecycle_state is ExecutionContinuationLifecycleState.RESUMED
    assert resumed.identity == _identity()
    assert resumed.pause_id == f"pause_{_C2}"
    assert resumed.human_request_id == f"hr_{_C2}"


@pytest.mark.parametrize(
    "setup",
    [
        "resume_authorized",
        "resumed",
        "pause_requested",
        "paused",
        "rejected",
        "escalated",
        "cancelled",
    ],
)
def test_lifecycle_preserved_after_restart(setup: str) -> None:
    proc_a = _spawn_process_a()
    if setup == "resumed":
        pending = _drive_to_resumed(proc_a.service, proc_a.driver, _C2)
    elif setup == "pause_requested":
        pending = proc_a.service.request_pause(_pause_request(_C2))
    elif setup == "paused":
        proc_a.service.request_pause(_pause_request(_C2))
        pending = proc_a.driver.record_execution_reached_safe_pause(
            _C2,
            execution_pause_established=True,
        )
    elif setup == "resume_authorized":
        waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
        pending = proc_a.service.apply_resolution(
            _resolution(_C2, expected_revision=waiting.revision),
        )
    elif setup == "rejected":
        waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
        pending = proc_a.service.apply_resolution(
            _resolution(
                _C2,
                expected_revision=waiting.revision,
                verdict=ExecutionHumanVerdict.REJECT,
            ),
        )
    elif setup == "escalated":
        waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
        pending = proc_a.service.apply_resolution(
            _resolution(
                _C2,
                expected_revision=waiting.revision,
                verdict=ExecutionHumanVerdict.ESCALATE,
            ),
        )
    else:
        waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
        cancelled_state = advance_continuation_lifecycle(
            waiting.lifecycle_state,
            ExecutionContinuationTransition.CANCEL_CONTINUATION,
        )
        pending = waiting.model_copy(
            update={"lifecycle_state": cancelled_state, "revision": waiting.revision + 1},
        )
        assert proc_a.store.compare_and_swap(
            continuation_id=_C2,
            expected=waiting,
            updated=pending,
        )
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    assert qual.current_episode.continuation_id == _C2
    assert qual.current_episode.lifecycle_state is pending.lifecycle_state
    assert qual.current_episode.revision == pending.revision
    blocks = setup not in {"resumed", "pause_requested"}
    if blocks:
        with pytest.raises(ExecutionContinuationError):
            assert_canonical_execution_may_progress(store=proc_b.store, identity=qual.identity)
    else:
        assert_canonical_execution_may_progress(store=proc_b.store, identity=qual.identity)


def test_history_c1_c2_after_restart() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    assert proc_b.store.load(_C1) is not None
    assert proc_b.store.load(_C2) is not None


def test_c3_after_restart_becomes_current() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    _drive_to_resumed(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    _recover(proc_b.store, _C2)
    _drive_to_waiting(proc_b.service, proc_b.driver, _C3)
    current = proc_b.store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C3


def test_stale_c1_human_input_after_restart() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    waiting_c2 = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    with pytest.raises(ExecutionContinuationError) as exc:
        proc_b.service.apply_resolution(
            _resolution(_C1, expected_revision=1),
        )
    assert exc.value.code in {
        ExecutionContinuationErrorCode.STALE_REVISION,
        ExecutionContinuationErrorCode.INVALID_TRANSITION,
        ExecutionContinuationErrorCode.ALREADY_RESOLVED,
    }
    approved = proc_b.service.apply_resolution(
        _resolution(_C2, expected_revision=waiting_c2.revision),
    )
    assert approved.lifecycle_state is ExecutionContinuationLifecycleState.RESUME_AUTHORIZED
    assert qual.identity == _identity()


def test_task_projection_absent_canonical_blocks() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    with pytest.raises(ExecutionContinuationError):
        assert_canonical_execution_may_progress(store=proc_b.store, identity=qual.identity)


def test_task_projection_corrupted_canonical_blocks() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    task = _task()
    task.runtime.governance.projected_continuation_lifecycle_state = "RESUMED"
    with pytest.raises(ExecutionContinuationError):
        assert_canonical_execution_may_progress(store=proc_b.store, identity=qual.identity)


def test_reprojection_and_digest_deterministic() -> None:
    proc_a = _spawn_process_a()
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    digest_a = execution_continuation_projection_payload_digest(waiting)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    task = _task()
    sink = wire_task_execution_continuation_projection_sink(task)
    HumanPauseCoordinator.project_continuation(task, qual.current_episode, projection_sink=sink)
    digest_b = execution_continuation_projection_payload_digest(qual.current_episode)
    assert digest_a == digest_b
    assert task.runtime.governance.projected_continuation_payload_digest == digest_a


@pytest.mark.asyncio
async def test_runtime_recreation_boundary_blocks() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)

    class _Delegate:
        async def execute(self, request: object) -> str:
            return "ok"

    runtime_b = ExecutionRuntime(
        _Delegate(),
        continuation_state_store=proc_b.store,
    )
    assert runtime_b is not None
    boundary = ExecutionBoundary(
        _Delegate(),
        identity=qual.execution_identity_binding,
        authority=_AUTHORITY,
        continuation_state_store=proc_b.store,
    )
    with pytest.raises(ExecutionContinuationError):
        await boundary.execute("probe")


def test_no_new_attempt_or_execution_id_on_qualify() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    assert qual.identity.attempt_id == _ATTEMPT
    assert qual.identity.execution_id == _EXECUTION
    assert qual.identity.run_id == _RUN
    assert qual.identity.task_id == _TASK


def test_non_durable_restore_rejected() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    _drive_to_waiting(service, ExecutionContinuationLifecycleDriver(service), _C2)
    with pytest.raises(ExecutionContinuationError) as exc:
        recover_execution_continuation_process_restart(
            store=store,
            recovery_handle=execution_continuation_recovery_handle_for_continuation_id(_C2),
            authority=_AUTHORITY,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_non_durable_normal_execution_still_works() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    waiting = _drive_to_waiting(deps.continuation_service, deps.lifecycle_driver, _C2)
    assert waiting.continuation_id == _C2
    assert store.is_durable is False
    with pytest.raises(ExecutionContinuationError) as exc:
        qualify_execution_continuation_process_restart(
            store=store,
            identity=_identity(),
            authority=_AUTHORITY,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_pointer_to_missing_record_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    payload["current_by_identity"][0]["continuation_id"] = "missing-episode"
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing(payload)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_pointer_identity_mismatch_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    payload["records"][_C2]["identity"]["attempt_id"] = str(mint_attempt_id())
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing(payload)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_stale_cas_after_restart() -> None:
    proc_a = _spawn_process_a()
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    with pytest.raises(ExecutionContinuationError) as exc:
        proc_b.service.apply_resolution(
            _resolution(_C2, expected_revision=waiting.revision - 1),
        )
    assert exc.value.code is ExecutionContinuationErrorCode.STALE_REVISION


def test_double_resume_after_restart_one_wins() -> None:
    proc_a = _spawn_process_a()
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    approved = proc_a.service.apply_resolution(
        _resolution(_C2, expected_revision=waiting.revision),
    )
    proc_b = _true_restart_process_b(proc_a)
    barrier = threading.Barrier(2)
    results: list[ExecutionContinuationErrorCode | PendingExecutionContinuation] = []

    def _attempt() -> None:
        local_store = backing_execution_continuation_state_store(proc_b.backing)
        local = ExecutionContinuationService(local_store)
        barrier.wait()
        try:
            out = local.resume(_resume(_C2, expected_revision=approved.revision))
            results.append(out)
        except ExecutionContinuationError as exc:
            results.append(exc.code)

    t1 = threading.Thread(target=_attempt)
    t2 = threading.Thread(target=_attempt)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    successes = [r for r in results if isinstance(r, PendingExecutionContinuation)]
    assert len(successes) == 1


class _CustomDurableContinuationStore(ExecutionContinuationStateStore):
    def __init__(self, shared: dict[str, PendingExecutionContinuation], current: dict) -> None:
        self._records = shared
        self._current = current
        self._lock = threading.Lock()

    @property
    def is_durable(self) -> bool:
        return True

    def load(self, continuation_id: str) -> PendingExecutionContinuation | None:
        with self._lock:
            return self._records.get(continuation_id)

    def find_by_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        with self._lock:
            matches = [p for p in self._records.values() if p.identity == identity]
        return matches[0] if len(matches) == 1 else None

    def resolve_current_episode_for_identity(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        key = (
            str(identity.task_id),
            str(identity.run_id),
            str(identity.attempt_id),
            str(identity.execution_id),
        )
        with self._lock:
            cid = self._current.get(key)
            if cid is None:
                return None
            return self._records.get(cid)

    def resolve_identity_for_execution_progress(
        self,
        identity: ExecutionContinuationIdentity,
    ) -> PendingExecutionContinuation | None:
        return self.resolve_current_episode_for_identity(identity)

    def begin_current_episode_if_predecessor_allows(
        self,
        pending: PendingExecutionContinuation,
    ) -> bool:
        key = (
            str(pending.identity.task_id),
            str(pending.identity.run_id),
            str(pending.identity.attempt_id),
            str(pending.identity.execution_id),
        )
        with self._lock:
            if pending.continuation_id in self._records:
                return False
            self._records[pending.continuation_id] = pending
            self._current[key] = pending.continuation_id
            return True

    def insert_if_absent(self, pending: PendingExecutionContinuation) -> bool:
        with self._lock:
            if pending.continuation_id in self._records:
                return False
            self._records[pending.continuation_id] = pending
            return True

    def compare_and_swap(
        self,
        *,
        continuation_id: str,
        expected: PendingExecutionContinuation,
        updated: PendingExecutionContinuation,
    ) -> bool:
        with self._lock:
            cur = self._records.get(continuation_id)
            if cur is None or cur != expected:
                return False
            self._records[continuation_id] = updated
            return True


def test_custom_durable_provider_qualification() -> None:
    records: dict[str, PendingExecutionContinuation] = {}
    current: dict[tuple[str, str, str, str], str] = {}
    store = _CustomDurableContinuationStore(records, current)
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    _drive_to_waiting(deps.continuation_service, deps.lifecycle_driver, _C2)
    qual = recover_execution_continuation_process_restart(
        store=store,
        recovery_handle=execution_continuation_recovery_handle_for_continuation_id(_C2),
        authority=_AUTHORITY,
    )
    assert qual.current_episode.continuation_id == _C2
    assert qual.identity == _identity()


class _FalseyDurableStore(_CustomDurableContinuationStore):
    @property
    def is_durable(self) -> bool:
        return False  # type: ignore[return-value]


def test_falsey_durable_provider_preserved() -> None:
    records: dict[str, PendingExecutionContinuation] = {}
    current: dict[tuple[str, str, str, str], str] = {}
    store = _FalseyDurableStore(records, current)
    deps = wire_execution_engine_continuation_dependencies(state_store=store)
    _drive_to_waiting(deps.continuation_service, deps.lifecycle_driver, _C2)
    assert not store.is_durable
    with pytest.raises(ExecutionContinuationError) as exc:
        recover_execution_continuation_process_restart(
            store=store,
            recovery_handle=execution_continuation_recovery_handle_for_continuation_id(_C2),
            authority=_AUTHORITY,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_unknown_schema_fail_closed() -> None:
    with pytest.raises(ValidationError):
        PendingExecutionContinuation.model_validate(
            {
                "schema_version": "pending_execution_continuation.v999",
                "continuation_id": _C2,
                "identity": {
                    "task_id": str(_TASK),
                    "run_id": str(_RUN),
                    "attempt_id": str(_ATTEMPT),
                    "execution_id": str(_EXECUTION),
                },
                "lifecycle_state": "waiting_for_human",
                "revision": 1,
                "reason": "security",
            },
        )


def test_checkpoint_identity_mismatch_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    bad = ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=mint_attempt_id(),
        execution_id=_EXECUTION,
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        recover_execution_continuation_process_restart(
            store=proc_b.store,
            recovery_handle=execution_continuation_recovery_handle_for_continuation_id(_C2),
            authority=_AUTHORITY,
            checkpoint_identity=bad,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.IDENTITY_MISMATCH


def test_restart_qualify_no_root_admission_on_resume() -> None:
    proc_a = _spawn_process_a()
    waiting = _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C2)
    approved = proc_b.service.apply_resolution(
        _resolution(_C2, expected_revision=waiting.revision),
    )
    proc_b.service.resume(_resume(_C2, expected_revision=approved.revision))
    assert qual.identity.execution_id == _EXECUTION


def test_restart_module_no_mint_ids() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "continuation" / "restart_qualification.py"
    source = path.read_text(encoding="utf-8")
    assert "mint_attempt_id" not in source
    assert "mint_execution_id" not in source
    assert "new_root_execution" not in source


def test_restart_module_no_nexus_imports() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "continuation" / "restart_qualification.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    assert not any("nexus" in m for m in modules)


def test_reconnect_shared_backing_is_not_restart_durable() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    reconnect_store = backing_execution_continuation_state_store(proc_a.backing)
    assert reconnect_store.is_durable is False
    assert reconnect_store.load(_C2) is not None
    with pytest.raises(ExecutionContinuationError) as exc:
        recover_execution_continuation_process_restart(
            store=reconnect_store,
            recovery_handle=execution_continuation_recovery_handle_for_continuation_id(_C2),
            authority=_AUTHORITY,
        )
    assert exc.value.code is ExecutionContinuationErrorCode.NON_DURABLE_CONTINUATION_STORE


def test_historical_recovery_handle_rejected_when_not_current() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    _drive_to_resumed(proc_a.service, proc_a.driver, _C2)
    _drive_to_waiting(proc_a.service, proc_a.driver, _C3)
    proc_b = _true_restart_process_b(proc_a)
    with pytest.raises(ExecutionContinuationError) as exc:
        _recover(proc_b.store, _C1)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_three_episode_history_roundtrip_after_true_restart() -> None:
    proc_a = _spawn_process_a()
    _drive_to_resumed(proc_a.service, proc_a.driver, _C1)
    _drive_to_resumed(proc_a.service, proc_a.driver, _C2)
    waiting_c3 = _drive_to_waiting(proc_a.service, proc_a.driver, _C3)
    proc_b = _true_restart_process_b(proc_a)
    qual = _recover(proc_b.store, _C3)
    assert qual.current_episode.continuation_id == _C3
    assert qual.current_episode.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    assert qual.current_episode.revision == waiting_c3.revision
    assert proc_b.store.load(_C1) is not None
    assert proc_b.store.load(_C2) is not None


def test_durable_export_unknown_schema_fail_closed() -> None:
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing({"schema_version": "execution_continuation_durable_state.v999"})
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_durable_export_missing_identity_field_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    record = payload["records"][_C2]
    del record["identity"]["execution_id"]
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing(payload)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_durable_export_invalid_lifecycle_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    payload["records"][_C2]["lifecycle_state"] = "not_a_real_state"
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing(payload)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_durable_export_invalid_revision_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    payload["records"][_C2]["revision"] = 0
    with pytest.raises(ExecutionContinuationError) as exc:
        restore_durable_continuation_backing(payload)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE


def test_export_snapshot_is_not_mutable_alias_of_backing() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    payload = export_durable_continuation_state(proc_a.backing)
    payload["records"][_C2]["revision"] = 999
    assert proc_a.store.load(_C2) is not None
    assert proc_a.store.load(_C2).revision != 999


def test_recovery_handle_not_found_fail_closed() -> None:
    proc_a = _spawn_process_a()
    _drive_to_waiting(proc_a.service, proc_a.driver, _C2)
    proc_b = _true_restart_process_b(proc_a)
    with pytest.raises(ExecutionContinuationError) as exc:
        _recover(proc_b.store, "missing-handle")
    assert exc.value.code is ExecutionContinuationErrorCode.NOT_FOUND


def test_restart_module_no_reference_backing_imports() -> None:
    path = _REPO_ROOT / "intergrax" / "runtime" / "execution" / "continuation" / "restart_qualification.py"
    source = path.read_text(encoding="utf-8")
    assert "persistence" not in source
    assert "BackingExecutionContinuationStateStore" not in source
