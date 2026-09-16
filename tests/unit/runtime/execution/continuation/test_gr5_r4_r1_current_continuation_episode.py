# © Artur Czarnecki. All rights reserved.

"""GR-5-R4-R1 — canonical current continuation episode resolution."""

from __future__ import annotations

import ast
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
    ExecutionContinuationResumeCommand,
    ExecutionHumanVerdict,
    ExecutionPauseRequest,
    PendingExecutionContinuation,
)
from intergrax.contracts.execution_continuation_state_store import ExecutionContinuationStateStore
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.human_approver import local_development_approver_evidence
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
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
    load_pending_for_execution_progress,
)
from intergrax.runtime.execution.continuation.service import ExecutionContinuationService
from intergrax.runtime.execution.runtime import ExecutionRuntime, RootExecutionContext
from intergrax.runtime.human.pause import HumanPauseCoordinator
from intergrax.runtime.nexus.orchestration.internal_continuation_orchestration import (
    InternalOrchestrationContinuation,
    canonical_allows_planning_progress_after_human_gate,
    canonical_execution_is_resumed,
)
from intergrax.runtime.task.task import Task
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SEED = "gr5-r4-r1"
_TASK = canonical_task_id_for_tests(_SEED)
_RUN = canonical_run_id_for_tests(_SEED)
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_C1 = "cont_r4r1_c1"
_C2 = "cont_r4r1_c2"
_C3 = "cont_r4r1_c3"
_AUTHORITY = ParentExecutionAuthority.unrestricted_root()


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
            actor_id="op-r4r1",
            tenant_id="tenant-gr5-r4-r1",
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


def _capability(
    deps: object | None = None,
) -> InternalOrchestrationContinuation:
    wired = deps or wire_execution_engine_continuation_dependencies()
    return InternalOrchestrationContinuation(
        port=wired.continuation,
        lifecycle_driver=wired.lifecycle_driver,
    )


def _deps_with_store(
    store: InMemoryExecutionContinuationStateStore,
) -> object:
    return wire_execution_engine_continuation_dependencies(state_store=store)


class _IndependentCurrentEpisodeStore(ExecutionContinuationStateStore):
    """Minimal alternate provider — not a subclass of the in-memory reference."""

    def __init__(self) -> None:
        self._records: dict[str, PendingExecutionContinuation] = {}
        self._current: dict[tuple[str, str, str, str], str] = {}
        self._lock = threading.Lock()

    @property
    def is_durable(self) -> bool:
        return False

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
            current_id = self._current.get(key)
            if current_id is None:
                return None
            return self._records.get(current_id)

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
            current_id = self._current.get(key)
            if current_id is not None:
                predecessor = self._records[current_id]
                if predecessor.lifecycle_state not in {
                    ExecutionContinuationLifecycleState.RESUMED,
                    ExecutionContinuationLifecycleState.REJECTED,
                    ExecutionContinuationLifecycleState.ESCALATED,
                    ExecutionContinuationLifecycleState.CANCELLED,
                }:
                    raise ExecutionContinuationError(
                        "predecessor not terminal",
                        code=ExecutionContinuationErrorCode.INVALID_TRANSITION,
                    )
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
            current = self._records.get(continuation_id)
            if current is None or current != expected:
                return False
            self._records[continuation_id] = updated
            return True


def test_c1_resumed_then_c2_waiting_current_is_c2() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None
    assert current.continuation_id == _C2
    assert current.lifecycle_state is ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN


def test_progress_gate_blocks_on_c2_waiting_after_c1_resumed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_canonical_execution_may_progress(store=store, identity=_identity())
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED


@pytest.mark.asyncio
async def test_boundary_blocks_when_c2_waiting() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)

    class _Delegate:
        async def execute(self, request: object) -> str:
            return "ok"

    boundary = ExecutionBoundary(
        _Delegate(),
        identity=ExecutionIdentityBinding(
            task_id=_TASK,
            run_id=_RUN,
            attempt_id=_ATTEMPT,
            execution_id=_EXECUTION,
        ),
        authority=_AUTHORITY,
        continuation_state_store=store,
    )
    with pytest.raises(ExecutionContinuationError):
        await boundary.execute("probe")


def test_c2_resumed_allows_progress() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    _drive_to_resumed(service, driver, _C2)
    assert_canonical_execution_may_progress(store=store, identity=_identity())


def test_c3_waiting_blocks_again() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    for cid in (_C1, _C2):
        _drive_to_resumed(service, driver, cid)
    _drive_to_waiting(service, driver, _C3)
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C3
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_canonical_execution_may_progress(store=store, identity=_identity())
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED


def test_high_revision_c1_low_revision_c3_current_is_c3() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    for _ in range(99):
        pending = service.get_pending(ExecutionContinuationLookup(continuation_id=_C1))
        service.store.compare_and_swap(
            continuation_id=_C1,
            expected=pending,
            updated=pending.model_copy(update={"revision": pending.revision + 1}),
        )
    _drive_to_resumed(service, driver, _C2)
    _drive_to_waiting(service, driver, _C3)
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C3


@pytest.mark.parametrize(
    "verdict",
    [ExecutionHumanVerdict.REJECT, ExecutionHumanVerdict.ESCALATE],
)
def test_terminal_current_episode_blocks_progress(verdict: ExecutionHumanVerdict) -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    waiting = _drive_to_waiting(service, driver, _C2)
    service.apply_resolution(
        _resolution(_C2, expected_revision=waiting.revision, verdict=verdict),
    )
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C2
    with pytest.raises(ExecutionContinuationError) as exc:
        assert_canonical_execution_may_progress(store=store, identity=_identity())
    assert exc.value.code is ExecutionContinuationErrorCode.EXECUTION_PROGRESS_BLOCKED


def test_two_active_episodes_fail_closed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    service = ExecutionContinuationService(store)
    service.request_pause(_pause_request(_C1))
    store.insert_if_absent(
        PendingExecutionContinuation(
            continuation_id=_C2,
            identity=_identity(),
            lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
            revision=1,
            reason=ContinuationReason.SECURITY,
        ),
    )
    with pytest.raises(ExecutionContinuationError) as exc:
        store.resolve_current_episode_for_identity(_identity())
    assert exc.value.code is ExecutionContinuationErrorCode.AMBIGUOUS_IDENTITY


def test_concurrent_successor_creation_one_wins() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    barrier = threading.Barrier(2)
    results: list[ExecutionContinuationErrorCode | None] = []

    def _attempt(cont_id: str) -> None:
        barrier.wait()
        local = ExecutionContinuationService(store)
        try:
            local.request_pause(_pause_request(cont_id))
            results.append(None)
        except ExecutionContinuationError as exc:
            results.append(exc.code)

    t1 = threading.Thread(target=_attempt, args=(_C2,))
    t2 = threading.Thread(target=_attempt, args=(_C3,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count(None) == 1
    assert ExecutionContinuationErrorCode.INVALID_TRANSITION in results
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None
    assert current.continuation_id in {_C2, _C3}


def test_failed_successor_leaves_current_unchanged() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_waiting(service, driver, _C1)
    with pytest.raises(ExecutionContinuationError) as exc:
        service.request_pause(_pause_request(_C2))
    assert exc.value.code is ExecutionContinuationErrorCode.INVALID_TRANSITION
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C1


def test_historical_load_by_continuation_id() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    for cid in (_C1, _C2, _C3):
        _drive_to_resumed(service, driver, cid)
    for cid in (_C1, _C2, _C3):
        assert store.load(cid) is not None


def test_no_current_episode_progress_allowed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    assert_canonical_execution_may_progress(store=store, identity=_identity())


def test_store_query_failure_fails_closed() -> None:
    class _FailingStore(InMemoryExecutionContinuationStateStore):
        def resolve_identity_for_execution_progress(
            self,
            identity: ExecutionContinuationIdentity,
        ) -> PendingExecutionContinuation | None:
            raise RuntimeError("unavailable")

    store = _FailingStore()
    with pytest.raises(ExecutionContinuationError) as exc:
        load_pending_for_execution_progress(store=store, identity=_identity())
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED


def test_r4_canonical_execution_is_resumed_after_c1_c2() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    cap = _capability(deps)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    identity = _identity()
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)
    assert canonical_execution_is_resumed(cap, identity=identity) is False
    assert (
        canonical_allows_planning_progress_after_human_gate(
            cap,
            identity=identity,
            human_approval_required=True,
        )
        is False
    )
    waiting_c2 = service.get_pending(ExecutionContinuationLookup(continuation_id=_C2))
    approved = service.apply_resolution(
        _resolution(_C2, expected_revision=waiting_c2.revision),
    )
    service.resume(_resume(_C2, expected_revision=approved.revision))
    assert canonical_execution_is_resumed(cap, identity=identity) is True
    assert (
        canonical_allows_planning_progress_after_human_gate(
            cap,
            identity=identity,
            human_approval_required=True,
        )
        is True
    )


def test_task_corruption_does_not_unblock_progress() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    waiting = _drive_to_waiting(service, driver, _C2)
    task = Task(tenant_id="t1", user_id="u1", message="m", task_id=_TASK)
    task.runtime.governance.paused = False
    HumanPauseCoordinator.project_continuation(task, waiting)
    with pytest.raises(ExecutionContinuationError):
        assert_canonical_execution_may_progress(store=store, identity=_identity())


def test_c2_projection_after_c1_resumed() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    task = Task(tenant_id="t1", user_id="u1", message="m", task_id=_TASK)
    c1_resumed = _drive_to_resumed(service, driver, _C1)
    HumanPauseCoordinator.project_continuation(task, c1_resumed)
    c2_waiting = _drive_to_waiting(service, driver, _C2)
    result = HumanPauseCoordinator.project_continuation(task, c2_waiting)
    assert result is not None


def test_new_service_instance_same_store() -> None:
    store = InMemoryExecutionContinuationStateStore()
    deps = _deps_with_store(store)
    service = deps.continuation_service
    driver = deps.lifecycle_driver
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)
    other = ExecutionContinuationService(store)
    pending = other.get_pending(ExecutionContinuationLookup(identity=_identity()))
    assert pending.continuation_id == _C2


def test_custom_independent_store_contract() -> None:
    store = _IndependentCurrentEpisodeStore()
    service = ExecutionContinuationService(store)
    driver = ExecutionContinuationLifecycleDriver(service)
    _drive_to_resumed(service, driver, _C1)
    _drive_to_waiting(service, driver, _C2)
    current = store.resolve_current_episode_for_identity(_identity())
    assert current is not None and current.continuation_id == _C2


def test_progress_gate_no_nexus_or_task_projection_authority() -> None:
    for rel in (
        "intergrax/runtime/execution/continuation/progress_gate.py",
        "intergrax/runtime/execution/continuation/service.py",
    ):
        source = (_REPO_ROOT / rel).read_text(encoding="utf-8")
        assert "projected_continuation_id" not in source
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert "runtime.nexus" not in node.module
