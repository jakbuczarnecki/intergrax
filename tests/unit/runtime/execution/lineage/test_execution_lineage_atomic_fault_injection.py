# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageUnavailableError,
    build_execution_lineage_attempt_scope,
)
from intergrax.runtime.execution.lineage.persistence import (
    _ExecutionLineageStoreLogic,
    _InMemoryPartitionAtomicRowStore,
    _PartitionAtomicRowBatch,
    _PartitionAtomicRowBatchResult,
    _PartitionReplaceIfMatchOnCreated,
)


def _scope() -> object:
    return build_execution_lineage_attempt_scope(
        tenant_id="tenant-a",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
    )


class _FaultInjectingPartitionStore(_InMemoryPartitionAtomicRowStore):
    def __init__(self) -> None:
        super().__init__()
        self.fail_primary_prefix: str | None = None

    def execute_partition_atomic_batch(
        self,
        batch: _PartitionAtomicRowBatch,
    ) -> _PartitionAtomicRowBatchResult:
        should_fail = (
            self.fail_primary_prefix is not None
            and batch.primary_put_if_absent.row_key.startswith(self.fail_primary_prefix)
        )
        if not should_fail:
            return super().execute_partition_atomic_batch(batch)
        with self._lock:
            snapshot = dict(self._rows)
            primary_created = self._put_if_absent_unlocked(
                batch.primary_put_if_absent,
                rows=snapshot,
            )
            if primary_created:
                for op in batch.on_created_ops:
                    if isinstance(op, _PartitionReplaceIfMatchOnCreated):
                        raise RuntimeError("forced on_created failure")
                    raise RuntimeError("forced on_created failure")
            self._rows = snapshot
            return _PartitionAtomicRowBatchResult(primary_created=primary_created)


def _logic(
    store: _FaultInjectingPartitionStore | None = None,
) -> _ExecutionLineageStoreLogic:
    return _ExecutionLineageStoreLogic(store or _FaultInjectingPartitionStore())


def test_a1_admission_conflict_leaves_no_partial_admission() -> None:
    scope = _scope()
    root = mint_execution_id()
    child = mint_execution_id()
    store = _FaultInjectingPartitionStore()
    logic = _logic(store)
    logic.open_attempt(scope)
    logic.open_segment(scope, root)
    logic.admit_root(scope, root, root)
    store.fail_primary_prefix = "admission:"
    with pytest.raises(ExecutionLineageUnavailableError):
        logic.admit_child(scope, root, child, root)
    state = logic.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.next_admission_position == 2
    page = logic.list_admissions_for_attempt(scope, limit=10)
    assert len(page.admissions) == 1


def test_a2_segment_open_conflict_leaves_no_partial_segment() -> None:
    scope = _scope()
    root = mint_execution_id()
    store = _FaultInjectingPartitionStore()
    logic = _logic(store)
    logic.open_attempt(scope)
    store.fail_primary_prefix = "segment:"
    with pytest.raises(ExecutionLineageUnavailableError):
        logic.open_segment(scope, root)
    state = logic.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.active_segment_root_execution_id is None


def test_a3_predecessor_unclean_conflict_is_all_or_nothing() -> None:
    scope = _scope()
    e1 = mint_execution_id()
    e4 = mint_execution_id()
    store = _FaultInjectingPartitionStore()
    logic = _logic(store)
    logic.open_attempt(scope)
    logic.open_segment(scope, e1)
    store.fail_primary_prefix = "segment:"
    with pytest.raises(ExecutionLineageUnavailableError):
        logic.open_segment(scope, e4, e1)
    state = logic.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.active_segment_root_execution_id == e1
    assert state.degraded is False
    from intergrax.runtime.execution.lineage.persistence import (
        _segment_row_key,
        execution_lineage_partition_key,
    )

    partition = execution_lineage_partition_key(scope)
    predecessor = store.get_row(partition, _segment_row_key(e1))
    assert predecessor is not None
    successor = store.get_row(partition, _segment_row_key(e4))
    assert successor is None


def test_a4_seal_conflict_leaves_no_partial_seal() -> None:
    scope = _scope()
    root = mint_execution_id()
    store = _FaultInjectingPartitionStore()
    logic = _logic(store)
    logic.open_attempt(scope)
    logic.open_segment(scope, root)
    logic.admit_root(scope, root, root)
    store.fail_primary_prefix = "meta:seal"
    with pytest.raises(ExecutionLineageUnavailableError):
        logic.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    state = logic.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.sealed is False
    assert logic.read_seal(scope) is None


def test_a5_seal_with_segment_close_conflict_is_all_or_nothing() -> None:
    scope = _scope()
    root = mint_execution_id()
    store = _FaultInjectingPartitionStore()
    logic = _logic(store)
    logic.open_attempt(scope)
    logic.open_segment(scope, root)
    logic.admit_root(scope, root, root)
    store.fail_primary_prefix = "meta:seal"
    with pytest.raises(ExecutionLineageUnavailableError):
        logic.seal_attempt(scope, ExecutionLineageAttemptClosureKind.COMPLETED)
    state = logic.read_attempt_lineage_state(scope)
    assert state is not None
    assert state.sealed is False
    from intergrax.runtime.execution.lineage.persistence import (
        _segment_row_key,
        execution_lineage_partition_key,
    )

    partition = execution_lineage_partition_key(scope)
    segment_row = store.get_row(partition, _segment_row_key(root))
    assert segment_row is not None
