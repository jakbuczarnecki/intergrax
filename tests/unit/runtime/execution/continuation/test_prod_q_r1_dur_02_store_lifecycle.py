# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-DUR-02 — serialized-seal store lifecycle (composition authority)."""

from __future__ import annotations

import subprocess
import sys
import threading
import textwrap
from pathlib import Path

import pytest

from intergrax.contracts.execution_continuation import (
    ExecutionContinuationError,
    ExecutionContinuationErrorCode,
    ExecutionContinuationIdentity,
    ExecutionContinuationLifecycleState,
    PendingExecutionContinuation,
)
from intergrax.contracts.governed_continuation_correlation import ContinuationReason
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    open_serialized_seal_execution_continuation_state_store,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = [pytest.mark.unit]

_TASK = canonical_task_id_for_tests("dur-02-lifecycle")
_RUN = canonical_run_id_for_tests("dur-02-lifecycle")
_CONT = "cont_dur02_lifecycle"


def _pending(continuation_id: str = _CONT) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
        continuation_id=continuation_id,
        identity=ExecutionContinuationIdentity(
            task_id=_TASK,
            run_id=_RUN,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        ),
        lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
        revision=1,
        reason=ContinuationReason.SECURITY,
        pause_id="pause",
        human_request_id="hr",
    )


def test_close_blocks_subsequent_mutations(state_path: Path) -> None:
    store = open_serialized_seal_execution_continuation_state_store(
        state_path,
        missing_file_policy="empty",
    )
    assert store.begin_current_episode_if_predecessor_allows(_pending())
    store.close()
    store.close()
    with pytest.raises(ExecutionContinuationError) as exc:
        store.insert_if_absent(_pending(continuation_id="cont_two"))
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED


def test_close_waits_for_in_flight_mutation(state_path: Path) -> None:
    store = open_serialized_seal_execution_continuation_state_store(
        state_path,
        missing_file_policy="empty",
    )
    assert store.begin_current_episode_if_predecessor_allows(_pending())
    mutation_done = threading.Event()
    close_done = threading.Event()

    def slow_persist(_payload: object) -> None:
        close_done.wait(timeout=2.0)
        store._file_persistence.atomic_write_export_payload(_payload)  # type: ignore[arg-type]

    store._persist_export_payload = slow_persist  # type: ignore[method-assign]
    current = store.load(_CONT)
    assert current is not None
    updated = current.model_copy(update={"revision": 2})

    def mutate() -> None:
        store.compare_and_swap(
            continuation_id=_CONT,
            expected=current,
            updated=updated,
        )
        mutation_done.set()

    tm = threading.Thread(target=mutate)
    tc = threading.Thread(target=store.close)
    tm.start()
    tc.start()
    close_done.set()
    tm.join(timeout=3.0)
    tc.join(timeout=3.0)
    assert mutation_done.is_set()
    with pytest.raises(ExecutionContinuationError):
        store.compare_and_swap(
            continuation_id=_CONT,
            expected=updated,
            updated=updated.model_copy(update={"revision": 3}),
        )


def test_writer_lock_released_after_close_allows_reopen(state_path: Path) -> None:
    store = open_serialized_seal_execution_continuation_state_store(
        state_path,
        missing_file_policy="empty",
    )
    assert store.begin_current_episode_if_predecessor_allows(_pending())
    store.close()
    reopened = open_serialized_seal_execution_continuation_state_store(state_path)
    assert reopened.load(_CONT) is not None
    reopened.close()


def test_os_process_restart_reads_prior_sealed_state(tmp_path: Path) -> None:
    state_path = tmp_path / "continuation.json"
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
            open_serialized_seal_execution_continuation_state_store,
        )
        from intergrax.contracts.execution_continuation import (
            ExecutionContinuationIdentity,
            ExecutionContinuationLifecycleState,
            PendingExecutionContinuation,
        )
        from intergrax.contracts.governed_continuation_correlation import ContinuationReason
        from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id
        from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

        path = Path(sys.argv[1])
        mode = sys.argv[2]
        store = open_serialized_seal_execution_continuation_state_store(
            path,
            missing_file_policy="empty",
        )
        if mode == "write":
            pending = PendingExecutionContinuation(
                continuation_id="cont_restart",
                identity=ExecutionContinuationIdentity(
                    task_id=canonical_task_id_for_tests("dur-02-restart"),
                    run_id=canonical_run_id_for_tests("dur-02-restart"),
                    attempt_id=mint_attempt_id(),
                    execution_id=mint_execution_id(),
                ),
                lifecycle_state=ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN,
                revision=1,
                reason=ContinuationReason.SECURITY,
                pause_id="pause",
                human_request_id="hr",
            )
            assert store.begin_current_episode_if_predecessor_allows(pending)
            store.close()
        else:
            loaded = store.load("cont_restart")
            assert loaded is not None
            assert loaded.revision == 1
            store.close()
        """
    )
    subprocess.run(
        [sys.executable, "-c", script, str(state_path), "write"],
        check=True,
        cwd=str(Path(__file__).resolve().parents[5]),
    )
    subprocess.run(
        [sys.executable, "-c", script, str(state_path), "read"],
        check=True,
        cwd=str(Path(__file__).resolve().parents[5]),
    )


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "continuation.json"
