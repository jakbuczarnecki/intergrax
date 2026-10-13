# © Artur Czarnecki. All rights reserved.

"""PROD-Q-R1-DUR-01 — ADR-02-R3 NEG-SEAL qualification (focused slice)."""

from __future__ import annotations

import ast
import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

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
from intergrax.runtime.execution.continuation.durable_state_file import (
    AmbiguousDurablePersistError,
)
from intergrax.runtime.execution.continuation.durable_serialized_seal_store import (
    SerializedSealExecutionContinuationStateStore,
    open_serialized_seal_execution_continuation_state_store,
    reload_serialized_seal_store_from_disk,
)
from testing_support.builder import canonical_run_id_for_tests, canonical_task_id_for_tests

pytestmark = [pytest.mark.unit]

_TASK = canonical_task_id_for_tests("prod-q-seal")
_RUN = canonical_run_id_for_tests("prod-q-seal")
_ATTEMPT = mint_attempt_id()
_EXECUTION = mint_execution_id()
_CONT = "cont_prod_q_seal"


def _identity() -> ExecutionContinuationIdentity:
    return ExecutionContinuationIdentity(
        task_id=_TASK,
        run_id=_RUN,
        attempt_id=_ATTEMPT,
        execution_id=_EXECUTION,
    )


def _pending(
    *,
    revision: int = 1,
    lifecycle: ExecutionContinuationLifecycleState = (
        ExecutionContinuationLifecycleState.WAITING_FOR_HUMAN
    ),
    continuation_id: str = _CONT,
) -> PendingExecutionContinuation:
    return PendingExecutionContinuation(
        continuation_id=continuation_id,
        identity=_identity(),
        lifecycle_state=lifecycle,
        revision=revision,
        reason=ContinuationReason.SECURITY,
        pause_id="pause",
        human_request_id="hr",
    )


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "continuation.json"


def _open(path: Path, **kwargs: object) -> SerializedSealExecutionContinuationStateStore:
    return open_serialized_seal_execution_continuation_state_store(path, **kwargs)  # type: ignore[arg-type]


def test_neg_seal_01_persistence_failure_rolls_back_memory(state_path: Path) -> None:
    store = _open(state_path)

    def fail_persist(_payload: object) -> None:
        raise ExecutionContinuationError(
            "injected persist failure",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        )

    store._persist_export_payload = fail_persist  # type: ignore[method-assign]
    pending = _pending()
    with pytest.raises(ExecutionContinuationError) as exc:
        store.begin_current_episode_if_predecessor_allows(pending)
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    assert store.load(_CONT) is None
    assert not state_path.is_file()
    store.close()


def test_neg_seal_04_concurrent_cas_one_wins(state_path: Path) -> None:
    store = _open(state_path)
    base = _pending(revision=1)
    assert store.begin_current_episode_if_predecessor_allows(base)
    results: list[bool] = []
    barrier = threading.Barrier(2)

    def worker(revision: int) -> None:
        expected = store.load(_CONT)
        assert expected is not None
        assert expected.revision == 1
        updated = expected.model_copy(update={"revision": revision})
        barrier.wait()
        results.append(
            store.compare_and_swap(
                continuation_id=_CONT,
                expected=expected,
                updated=updated,
            ),
        )

    t1 = threading.Thread(target=worker, args=(2,))
    t2 = threading.Thread(target=worker, args=(3,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert results.count(True) == 1
    assert results.count(False) == 1
    loaded = store.load(_CONT)
    assert loaded is not None
    assert loaded.revision in {2, 3}
    on_disk = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk["records"][_CONT]["revision"] == loaded.revision
    store.close()


def test_neg_seal_08_seal_order_matches_commit_order(state_path: Path) -> None:
    store = _open(state_path)
    order: list[int] = []

    def track_persist(_payload: object) -> None:
        order.append(len(order) + 1)
        store._file_persistence.atomic_write_export_payload(_payload)  # type: ignore[arg-type]

    store._persist_export_payload = track_persist  # type: ignore[method-assign]
    assert store.begin_current_episode_if_predecessor_allows(_pending(revision=1))
    current = store.load(_CONT)
    assert current is not None
    store.compare_and_swap(
        continuation_id=_CONT,
        expected=current,
        updated=current.model_copy(update={"revision": 2}),
    )
    current = store.load(_CONT)
    assert current is not None
    store.compare_and_swap(
        continuation_id=_CONT,
        expected=current,
        updated=current.model_copy(update={"revision": 3}),
    )
    assert order == [1, 2, 3]
    assert json.loads(state_path.read_text(encoding="utf-8"))["records"][_CONT]["revision"] == 3
    store.close()


def test_neg_seal_09_reader_blocks_under_lock(state_path: Path) -> None:
    store = _open(state_path)
    started = threading.Event()
    finished = threading.Event()
    mid_revision: int | None = None

    def slow_persist(payload: object) -> None:
        started.set()
        time.sleep(0.2)
        store._file_persistence.atomic_write_export_payload(payload)  # type: ignore[arg-type]

    store._persist_export_payload = slow_persist  # type: ignore[method-assign]

    def mutate() -> None:
        store.begin_current_episode_if_predecessor_allows(_pending(revision=7))
        finished.set()

    threading.Thread(target=mutate, daemon=True).start()
    assert started.wait(timeout=2.0)

    def read() -> None:
        nonlocal mid_revision
        snap = store.load(_CONT)
        mid_revision = snap.revision if snap is not None else 0

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    time.sleep(0.05)
    assert not finished.is_set()
    assert not reader.is_alive() or mid_revision is None
    finished.wait(timeout=2.0)
    reader.join(timeout=2.0)
    assert store.load(_CONT) is not None
    store.close()


def test_neg_seal_12_ambiguous_persist_fail_closed(state_path: Path) -> None:
    store = _open(state_path)

    def ambiguous(_payload: object) -> None:
        raise AmbiguousDurablePersistError("simulated ambiguous io")

    store._persist_export_payload = ambiguous  # type: ignore[method-assign]
    with pytest.raises(ExecutionContinuationError) as exc:
        store.begin_current_episode_if_predecessor_allows(_pending())
    assert exc.value.code is ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED
    assert store.reconciliation_required
    with pytest.raises(ExecutionContinuationError) as exc2:
        store.insert_if_absent(_pending(continuation_id="other"))
    assert exc2.value.code is ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED
    store.close()


def test_neg_seal_13_no_nested_public_export_while_locked(state_path: Path) -> None:
    module_path = (
        Path(__file__).resolve().parents[3]
        / "intergrax"
        / "runtime"
        / "execution"
        / "continuation"
        / "durable_serialized_seal_store.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    source = ast.unparse(tree)
    assert "export_durable_continuation_state(" not in source
    assert "compare_and_swap_in_maps" in source
    assert "build_durable_export_payload_from_maps" in source


def test_insert_if_absent_sealed_mutation(state_path: Path) -> None:
    store = _open(state_path)
    alt = _pending(continuation_id="cont_alt")
    assert store.insert_if_absent(alt)
    assert store.insert_if_absent(alt) is False
    assert state_path.is_file()
    store.close()


def test_writer_lock_conflict_fail_closed(state_path: Path) -> None:
    first = _open(state_path)
    with pytest.raises(ExecutionContinuationError) as exc:
        _open(state_path)
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    first.close()
    second = _open(state_path)
    second.close()


def test_neg_seal_14_pre_replace_failure_safe_rollback(state_path: Path) -> None:
    store = _open(state_path)
    persistence = store._file_persistence
    original_write = persistence._atomic_write

    def fail_before_replace(_payload: object) -> None:
        raise ExecutionContinuationError(
            "injected pre-replace failure",
            code=ExecutionContinuationErrorCode.STORE_QUERY_FAILED,
        )

    persistence._atomic_write = fail_before_replace  # type: ignore[method-assign]
    pending = _pending()
    with pytest.raises(ExecutionContinuationError) as exc:
        store.begin_current_episode_if_predecessor_allows(pending)
    assert exc.value.code is ExecutionContinuationErrorCode.STORE_QUERY_FAILED
    assert store.load(_CONT) is None
    assert not store.reconciliation_required
    persistence._atomic_write = original_write  # type: ignore[method-assign]
    store.close()


def test_neg_seal_15_ambiguous_replace_reconciliation_no_rollback(
    state_path: Path,
) -> None:
    store = _open(state_path)
    assert store.begin_current_episode_if_predecessor_allows(_pending(revision=1))
    on_disk_before = json.loads(state_path.read_text(encoding="utf-8"))
    current = store.load(_CONT)
    assert current is not None

    real_replace = Path.replace

    def fail_replace(self: Path, target: Path) -> Path:
        if self == state_path.with_suffix(state_path.suffix + ".tmp"):
            raise OSError("simulated ambiguous replace")
        return real_replace(self, target)

    with patch.object(Path, "replace", fail_replace):
        with pytest.raises(ExecutionContinuationError) as exc:
            store.compare_and_swap(
                continuation_id=_CONT,
                expected=current,
                updated=current.model_copy(update={"revision": 2}),
            )
    assert exc.value.code is ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED
    assert store.reconciliation_required
    loaded = store.load(_CONT)
    assert loaded is not None
    assert loaded.revision == 2
    on_disk_after = json.loads(state_path.read_text(encoding="utf-8"))
    assert on_disk_after == on_disk_before
    store.close()


def test_neg_seal_16_concurrent_mutator_cannot_bypass_reconciliation(
    state_path: Path,
) -> None:
    store = _open(state_path)
    assert store.begin_current_episode_if_predecessor_allows(_pending(revision=1))
    persist_entered = threading.Event()
    release_persist = threading.Event()
    results: list[ExecutionContinuationErrorCode | None] = []

    def slow_ambiguous(_payload: object) -> None:
        persist_entered.set()
        assert release_persist.wait(timeout=2.0)
        raise AmbiguousDurablePersistError("injected ambiguous")

    store._persist_export_payload = slow_ambiguous  # type: ignore[method-assign]
    current = store.load(_CONT)
    assert current is not None

    def mutator_a() -> None:
        try:
            store.compare_and_swap(
                continuation_id=_CONT,
                expected=current,
                updated=current.model_copy(update={"revision": 99}),
            )
        except ExecutionContinuationError:
            pass

    def mutator_b() -> None:
        assert persist_entered.wait(timeout=2.0)
        try:
            store.insert_if_absent(_pending(continuation_id="cont_bypass"))
        except ExecutionContinuationError as exc:
            results.append(exc.code)

    ta = threading.Thread(target=mutator_a)
    tb = threading.Thread(target=mutator_b)
    ta.start()
    tb.start()
    release_persist.set()
    ta.join(timeout=3.0)
    tb.join(timeout=3.0)
    assert store.reconciliation_required
    assert results == [ExecutionContinuationErrorCode.RECONCILIATION_REQUIRED]
    store.close()


def test_neg_seal_17_reload_synchronized_with_mutations(state_path: Path) -> None:
    store = _open(state_path)
    assert store.begin_current_episode_if_predecessor_allows(_pending())

    def ambiguous(_payload: object) -> None:
        raise AmbiguousDurablePersistError("ambiguous")

    store._persist_export_payload = ambiguous  # type: ignore[method-assign]
    with pytest.raises(ExecutionContinuationError):
        store.begin_current_episode_if_predecessor_allows(
            _pending(continuation_id="cont_two"),
        )
    store._persist_export_payload = (
        store._file_persistence.atomic_write_export_payload
    )
    reload_started = threading.Event()
    reload_release = threading.Event()
    original_load = store._file_persistence.load_export_payload

    def slow_load() -> object:
        reload_started.set()
        assert reload_release.wait(timeout=2.0)
        return original_load()

    store._file_persistence.load_export_payload = slow_load  # type: ignore[method-assign]
    mutation_finished = threading.Event()

    def reload_worker() -> None:
        reload_serialized_seal_store_from_disk(store)

    def mutate_worker() -> None:
        reload_started.wait(timeout=2.0)
        try:
            store.insert_if_absent(_pending(continuation_id="cont_race"))
        except ExecutionContinuationError:
            pass
        mutation_finished.set()

    tr = threading.Thread(target=reload_worker)
    tm = threading.Thread(target=mutate_worker)
    tr.start()
    tm.start()
    time.sleep(0.05)
    assert not mutation_finished.is_set()
    reload_release.set()
    tr.join(timeout=3.0)
    tm.join(timeout=3.0)
    assert mutation_finished.is_set()
    assert not store.reconciliation_required
    store.close()


def test_neg_seal_18_corrupt_init_releases_writer_lock(state_path: Path) -> None:
    state_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(ExecutionContinuationError) as exc:
        _open(state_path)
    assert exc.value.code is ExecutionContinuationErrorCode.CORRUPT_CONTINUATION_STATE

    state_path.unlink()
    good = _open(state_path, missing_file_policy="empty")
    assert good.begin_current_episode_if_predecessor_allows(_pending())
    good.close()

    reopened = _open(state_path)
    assert reopened.load(_CONT) is not None
    reopened.close()


def test_reload_after_reconciliation(state_path: Path) -> None:
    store = _open(state_path)
    assert store.begin_current_episode_if_predecessor_allows(_pending())

    def ambiguous(_payload: object) -> None:
        raise AmbiguousDurablePersistError("ambiguous")

    store._persist_export_payload = ambiguous  # type: ignore[method-assign]
    with pytest.raises(ExecutionContinuationError):
        store.begin_current_episode_if_predecessor_allows(
            _pending(continuation_id="cont_two"),
        )
    reload_serialized_seal_store_from_disk(store)
    assert not store.reconciliation_required
    assert store.load(_CONT) is not None
    store.close()


# Deferred: NEG-SEAL-02 (OS new-process restart), NEG-SEAL-03 (corrupt file — existing UCA-6C),
# NEG-SEAL-05 (composition gate), NEG-SEAL-06, NEG-SEAL-07, NEG-SEAL-10, NEG-SEAL-11
