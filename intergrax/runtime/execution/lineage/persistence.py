# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution lineage persistence implementations (DG-001 R1)."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from intergrax.contracts.execution_identity import ExecutionId, validate_execution_id
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionPage,
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    ExecutionLineageSealRecord,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentRecord,
    ExecutionLineageUnavailableError,
    validate_admission_page_limit,
)
from intergrax.runtime.execution.lineage.codecs import (
    decode_execution_lineage_admission_record,
    decode_execution_lineage_attempt_state,
    decode_execution_lineage_segment_record,
    decode_execution_lineage_seal_record,
    encode_execution_lineage_admission_record,
    encode_execution_lineage_attempt_state,
    encode_execution_lineage_segment_record,
    encode_execution_lineage_seal_record,
)

_PARTITION_PREFIX = "intergrax.execution_lineage.v1"
_META_ROW = "meta:attempt"
_SEAL_ROW = "meta:seal"
_SEGMENT_ROW_PREFIX = "segment:"
_ADMISSION_ROW_PREFIX = "admission:"
_MAX_ATOMIC_RETRIES = 256


def execution_lineage_partition_key(scope: ExecutionLineageAttemptScope) -> str:
    return (
        f"{_PARTITION_PREFIX}:tenant:{scope.tenant_id}:task:{scope.task_id}:"
        f"run:{scope.run_id}:attempt:{scope.attempt_id}"
    )


def _segment_row_key(root_execution_id: ExecutionId) -> str:
    return f"{_SEGMENT_ROW_PREFIX}{root_execution_id}"


def _admission_row_key(execution_id: ExecutionId) -> str:
    return f"{_ADMISSION_ROW_PREFIX}{execution_id}"


def _scopes_match(left: ExecutionLineageAttemptScope, right: ExecutionLineageAttemptScope) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.task_id == right.task_id
        and left.run_id == right.run_id
        and left.attempt_id == right.attempt_id
    )


def _initial_attempt_state(scope: ExecutionLineageAttemptScope) -> ExecutionLineageAttemptState:
    return ExecutionLineageAttemptState(
        scope=scope,
        generation=1,
        next_admission_position=1,
        active_segment_root_execution_id=None,
        degraded=False,
        sealed=False,
        closure_kind=None,
    )


@dataclass(frozen=True, slots=True)
class _PartitionRow:
    partition_key: str
    row_key: str
    data: dict[str, object]


class _PartitionRowStore(Protocol):
    def get_row(self, partition_key: str, row_key: str) -> _PartitionRow | None:
        ...

    def put_if_absent(self, row: _PartitionRow) -> bool:
        ...

    def replace_if_match(self, expected: _PartitionRow, replacement: _PartitionRow) -> bool:
        ...

    def list_rows(
        self,
        partition_key: str,
        *,
        row_key_prefix: str,
        limit: int,
        cursor: str | None,
        sort_path: str,
    ) -> tuple[tuple[_PartitionRow, ...], str | None]:
        ...


class _ExecutionLineageStoreLogic:
    """Shared lineage semantics over a partition row store."""

    def __init__(self, store: _PartitionRowStore) -> None:
        self._store = store

    def open_attempt(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageAttemptState:
        partition = execution_lineage_partition_key(scope)
        existing = self._read_attempt_state(partition)
        if existing is not None:
            if not _scopes_match(existing.scope, scope):
                raise ExecutionLineageIntegrityError("attempt scope mismatch")
            return existing
        initial = _initial_attempt_state(scope)
        created = self._store.put_if_absent(
            _PartitionRow(partition, _META_ROW, encode_execution_lineage_attempt_state(initial)),
        )
        if created:
            return initial
        loaded = self._read_attempt_state(partition)
        if loaded is None:
            raise ExecutionLineageUnavailableError("attempt open race left no durable state")
        return loaded

    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        root = validate_execution_id(root_execution_id)
        predecessor = (
            validate_execution_id(predecessor_root_execution_id)
            if predecessor_root_execution_id is not None
            else None
        )
        if predecessor is not None and predecessor == root:
            raise ExecutionLineageIntegrityError("segment predecessor cannot equal current root")
        partition = execution_lineage_partition_key(scope)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            if attempt_state.sealed:
                raise ExecutionLineageIntegrityError("cannot open segment on sealed attempt")
            segment_key = _segment_row_key(root)
            existing_segment = self._read_segment(partition, segment_key)
            if existing_segment is not None:
                if not _scopes_match(existing_segment.scope, scope):
                    raise ExecutionLineageIntegrityError("segment scope mismatch")
                if existing_segment.root_execution_id != root:
                    raise ExecutionLineageIntegrityError("segment root mismatch")
                if existing_segment.predecessor_root_execution_id != predecessor:
                    raise ExecutionLineageIntegrityError("segment predecessor mismatch")
                return existing_segment
            if predecessor is None:
                if attempt_state.active_segment_root_execution_id is not None:
                    raise ExecutionLineageIntegrityError(
                        "concurrent initial segment without predecessor",
                    )
            else:
                self._validate_predecessor_chain(
                    partition,
                    scope,
                    predecessor=predecessor,
                    new_root=root,
                )
            new_segment = ExecutionLineageSegmentRecord(
                scope=scope,
                root_execution_id=root,
                predecessor_root_execution_id=predecessor,
                lifecycle=ExecutionLineageSegmentLifecycle.SEGMENT_OPEN,
            )
            updated_attempt = attempt_state.model_copy(
                update={
                    "generation": attempt_state.generation + 1,
                    "active_segment_root_execution_id": root,
                    "degraded": (
                        attempt_state.degraded
                        or self._requires_unclean_degradation(partition, predecessor)
                    ),
                },
            )
            if predecessor is not None:
                predecessor_segment = self._read_segment(partition, _segment_row_key(predecessor))
                if predecessor_segment is None:
                    raise ExecutionLineageIntegrityError("predecessor segment missing")
                if predecessor_segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN:
                    unclean_predecessor = predecessor_segment.model_copy(
                        update={"lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_UNCLEAN},
                    )
                    if not self._write_segment_and_attempt(
                        partition,
                        attempt_state,
                        updated_attempt,
                        segment_key,
                        new_segment,
                        predecessor_expected=predecessor_segment,
                        predecessor_replacement=unclean_predecessor,
                    ):
                        continue
                    return new_segment
            if self._write_segment_and_attempt(
                partition,
                attempt_state,
                updated_attempt,
                segment_key,
                new_segment,
            ):
                return new_segment
        raise ExecutionLineageUnavailableError("failed to open segment after bounded retries")

    def admit_root(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        segment_root = validate_execution_id(segment_root_execution_id)
        execution = validate_execution_id(execution_id)
        if execution != segment_root:
            raise ExecutionLineageIntegrityError("root admission execution_id mismatch")
        return self._admit(
            scope=scope,
            segment_root_execution_id=segment_root,
            execution_id=execution,
            parent_execution_id=None,
            graph_node_id=graph_node_id,
        )

    def admit_child(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        return self._admit(
            scope=scope,
            segment_root_execution_id=validate_execution_id(segment_root_execution_id),
            execution_id=validate_execution_id(execution_id),
            parent_execution_id=validate_execution_id(parent_execution_id),
            graph_node_id=graph_node_id,
        )

    def close_segment_for_resume(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
    ) -> ExecutionLineageSegmentRecord:
        partition = execution_lineage_partition_key(scope)
        root = validate_execution_id(root_execution_id)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            if attempt_state.sealed:
                raise ExecutionLineageIntegrityError("cannot close segment on sealed attempt")
            segment = self._read_segment(partition, _segment_row_key(root))
            if segment is None:
                raise ExecutionLineageIntegrityError("segment not found for clean close")
            if segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN:
                return segment
            if segment.lifecycle is not ExecutionLineageSegmentLifecycle.SEGMENT_OPEN:
                raise ExecutionLineageIntegrityError("segment not open for clean close")
            closed = segment.model_copy(
                update={"lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN},
            )
            if self._replace_segment(partition, segment, closed):
                return closed
        raise ExecutionLineageUnavailableError("failed to close segment after bounded retries")

    def mark_degraded(
        self,
        scope: ExecutionLineageAttemptScope,
        reason_code: str,
    ) -> ExecutionLineageAttemptState:
        del reason_code
        partition = execution_lineage_partition_key(scope)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            if attempt_state.degraded:
                return attempt_state
            updated = attempt_state.model_copy(update={"generation": attempt_state.generation + 1, "degraded": True})
            if self._replace_attempt_state(partition, attempt_state, updated):
                return updated
        raise ExecutionLineageUnavailableError("failed to mark degraded after bounded retries")

    def seal_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        closure_kind: ExecutionLineageAttemptClosureKind,
    ) -> ExecutionLineageSealRecord:
        partition = execution_lineage_partition_key(scope)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            existing_seal = self._read_seal(partition)
            if existing_seal is not None:
                if existing_seal.closure_kind != closure_kind:
                    raise ExecutionLineageIntegrityError("conflicting attempt seal")
                return existing_seal
            if attempt_state.sealed:
                existing = self._read_seal(partition)
                if existing is None:
                    raise ExecutionLineageIntegrityError("sealed attempt without seal record")
                return existing
            updated_attempt = attempt_state.model_copy(
                update={
                    "generation": attempt_state.generation + 1,
                    "sealed": True,
                    "closure_kind": closure_kind,
                },
            )
            seal = ExecutionLineageSealRecord(
                scope=scope,
                closure_kind=closure_kind,
                degraded=updated_attempt.degraded,
            )
            active_root = attempt_state.active_segment_root_execution_id
            if active_root is not None:
                segment = self._read_segment(partition, _segment_row_key(active_root))
                if segment is not None and segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN:
                    closed_segment = segment.model_copy(
                        update={"lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN},
                    )
                    if not self._seal_with_segment_close(
                        partition,
                        attempt_state,
                        updated_attempt,
                        segment,
                        closed_segment,
                        seal,
                    ):
                        continue
                    return seal
            if self._seal_attempt_only(partition, attempt_state, updated_attempt, seal):
                return seal
        raise ExecutionLineageUnavailableError("failed to seal attempt after bounded retries")

    def list_admissions_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAdmissionPage:
        validated_limit = validate_admission_page_limit(limit)
        partition = execution_lineage_partition_key(scope)
        rows, next_cursor = self._store.list_rows(
            partition,
            row_key_prefix=_ADMISSION_ROW_PREFIX,
            limit=validated_limit,
            cursor=cursor,
            sort_path="admission_position",
        )
        admissions = tuple(
            decode_execution_lineage_admission_record(row.data) for row in rows
        )
        admissions = tuple(
            sorted(admissions, key=lambda item: item.admission_position),
        )
        return ExecutionLineageAdmissionPage(admissions=admissions, next_cursor=next_cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._read_attempt_state(execution_lineage_partition_key(scope))

    def read_seal(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageSealRecord | None:
        return self._read_seal(execution_lineage_partition_key(scope))

    def _admit(
        self,
        *,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId | None,
        graph_node_id: str | None,
    ) -> ExecutionLineageAdmissionRecord:
        partition = execution_lineage_partition_key(scope)
        admission_key = _admission_row_key(execution_id)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            if attempt_state.sealed:
                raise ExecutionLineageIntegrityError("cannot admit on sealed attempt")
            if attempt_state.active_segment_root_execution_id != segment_root_execution_id:
                raise ExecutionLineageIntegrityError("admission segment root mismatch")
            existing = self._read_admission(partition, admission_key)
            if existing is not None:
                return self._verify_idempotent_admission(
                    existing,
                    scope=scope,
                    segment_root_execution_id=segment_root_execution_id,
                    execution_id=execution_id,
                    parent_execution_id=parent_execution_id,
                )
            if parent_execution_id is not None:
                parent_key = _admission_row_key(parent_execution_id)
                if self._read_admission(partition, parent_key) is None:
                    raise ExecutionLineageIntegrityError("parent admission missing")
            position = attempt_state.next_admission_position
            record = ExecutionLineageAdmissionRecord(
                scope=scope,
                segment_root_execution_id=segment_root_execution_id,
                execution_id=execution_id,
                parent_execution_id=parent_execution_id,
                admission_position=position,
                graph_node_id=graph_node_id,
            )
            updated_attempt = attempt_state.model_copy(
                update={
                    "generation": attempt_state.generation + 1,
                    "next_admission_position": position + 1,
                },
            )
            if self._write_admission(partition, attempt_state, updated_attempt, admission_key, record):
                return record
        raise ExecutionLineageUnavailableError("failed to admit after bounded retries")

    def _verify_idempotent_admission(
        self,
        existing: ExecutionLineageAdmissionRecord,
        *,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId | None,
    ) -> ExecutionLineageAdmissionRecord:
        if not _scopes_match(existing.scope, scope):
            raise ExecutionLineageIntegrityError("admission scope mismatch")
        if existing.segment_root_execution_id != segment_root_execution_id:
            raise ExecutionLineageIntegrityError("admission segment mismatch")
        if existing.execution_id != execution_id:
            raise ExecutionLineageIntegrityError("admission execution mismatch")
        if existing.parent_execution_id != parent_execution_id:
            raise ExecutionLineageIntegrityError("conflicting parent for admission")
        return existing

    def _requires_unclean_degradation(
        self,
        partition: str,
        predecessor: ExecutionId | None,
    ) -> bool:
        if predecessor is None:
            return False
        segment = self._read_segment(partition, _segment_row_key(predecessor))
        return (
            segment is not None
            and segment.lifecycle is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN
        )

    def _validate_predecessor_chain(
        self,
        partition: str,
        scope: ExecutionLineageAttemptScope,
        *,
        predecessor: ExecutionId,
        new_root: ExecutionId,
    ) -> None:
        predecessor_segment = self._read_segment(partition, _segment_row_key(predecessor))
        if predecessor_segment is None:
            raise ExecutionLineageIntegrityError("predecessor segment does not exist")
        if not _scopes_match(predecessor_segment.scope, scope):
            raise ExecutionLineageIntegrityError("cross-attempt predecessor")
        visited: set[ExecutionId] = {new_root}
        current: ExecutionId | None = predecessor
        while current is not None:
            if current in visited:
                raise ExecutionLineageIntegrityError("segment continuation cycle")
            visited.add(current)
            segment = self._read_segment(partition, _segment_row_key(current))
            if segment is None:
                raise ExecutionLineageIntegrityError("continuation segment missing")
            current = segment.predecessor_root_execution_id

    def _read_attempt_state(self, partition: str) -> ExecutionLineageAttemptState | None:
        row = self._store.get_row(partition, _META_ROW)
        if row is None:
            return None
        return decode_execution_lineage_attempt_state(row.data)

    def _require_attempt_state(
        self,
        partition: str,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState:
        state = self._read_attempt_state(partition)
        if state is None:
            raise ExecutionLineageIntegrityError("attempt not open")
        if not _scopes_match(state.scope, scope):
            raise ExecutionLineageIntegrityError("attempt scope mismatch")
        return state

    def _read_segment(self, partition: str, row_key: str) -> ExecutionLineageSegmentRecord | None:
        row = self._store.get_row(partition, row_key)
        if row is None:
            return None
        return decode_execution_lineage_segment_record(row.data)

    def _read_admission(self, partition: str, row_key: str) -> ExecutionLineageAdmissionRecord | None:
        row = self._store.get_row(partition, row_key)
        if row is None:
            return None
        return decode_execution_lineage_admission_record(row.data)

    def _read_seal(self, partition: str) -> ExecutionLineageSealRecord | None:
        row = self._store.get_row(partition, _SEAL_ROW)
        if row is None:
            return None
        return decode_execution_lineage_seal_record(row.data)

    def _replace_attempt_state(
        self,
        partition: str,
        expected: ExecutionLineageAttemptState,
        replacement: ExecutionLineageAttemptState,
    ) -> bool:
        return self._store.replace_if_match(
            _PartitionRow(partition, _META_ROW, encode_execution_lineage_attempt_state(expected)),
            _PartitionRow(partition, _META_ROW, encode_execution_lineage_attempt_state(replacement)),
        )

    def _replace_segment(
        self,
        partition: str,
        expected: ExecutionLineageSegmentRecord,
        replacement: ExecutionLineageSegmentRecord,
    ) -> bool:
        row_key = _segment_row_key(expected.root_execution_id)
        return self._store.replace_if_match(
            _PartitionRow(partition, row_key, encode_execution_lineage_segment_record(expected)),
            _PartitionRow(partition, row_key, encode_execution_lineage_segment_record(replacement)),
        )

    def _write_segment_and_attempt(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        segment_row_key: str,
        segment: ExecutionLineageSegmentRecord,
        *,
        predecessor_expected: ExecutionLineageSegmentRecord | None = None,
        predecessor_replacement: ExecutionLineageSegmentRecord | None = None,
    ) -> bool:
        if predecessor_expected is not None and predecessor_replacement is not None:
            if not self._replace_segment(
                partition,
                predecessor_expected,
                predecessor_replacement,
            ):
                return False
        if not self._store.put_if_absent(
            _PartitionRow(partition, segment_row_key, encode_execution_lineage_segment_record(segment)),
        ):
            return False
        return self._replace_attempt_state(partition, attempt_expected, attempt_replacement)

    def _write_admission(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        admission_row_key: str,
        record: ExecutionLineageAdmissionRecord,
    ) -> bool:
        if not self._store.put_if_absent(
            _PartitionRow(
                partition,
                admission_row_key,
                encode_execution_lineage_admission_record(record),
            ),
        ):
            return False
        return self._replace_attempt_state(partition, attempt_expected, attempt_replacement)

    def _seal_attempt_only(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        seal: ExecutionLineageSealRecord,
    ) -> bool:
        if not self._store.put_if_absent(
            _PartitionRow(partition, _SEAL_ROW, encode_execution_lineage_seal_record(seal)),
        ):
            return False
        return self._replace_attempt_state(partition, attempt_expected, attempt_replacement)

    def _seal_with_segment_close(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        segment_expected: ExecutionLineageSegmentRecord,
        segment_replacement: ExecutionLineageSegmentRecord,
        seal: ExecutionLineageSealRecord,
    ) -> bool:
        if not self._replace_segment(partition, segment_expected, segment_replacement):
            return False
        if not self._store.put_if_absent(
            _PartitionRow(partition, _SEAL_ROW, encode_execution_lineage_seal_record(seal)),
        ):
            return False
        return self._replace_attempt_state(partition, attempt_expected, attempt_replacement)


class _InMemoryPartitionRowStore:
    def __init__(self) -> None:
        self._rows: dict[tuple[str, str], _PartitionRow] = {}
        self._lock = threading.RLock()

    def get_row(self, partition_key: str, row_key: str) -> _PartitionRow | None:
        with self._lock:
            return self._rows.get((partition_key, row_key))

    def put_if_absent(self, row: _PartitionRow) -> bool:
        with self._lock:
            key = (row.partition_key, row.row_key)
            if key in self._rows:
                return False
            self._rows[key] = row
            return True

    def replace_if_match(self, expected: _PartitionRow, replacement: _PartitionRow) -> bool:
        with self._lock:
            current = self._rows.get((expected.partition_key, expected.row_key))
            if current is None or current.data != expected.data:
                return False
            self._rows[(replacement.partition_key, replacement.row_key)] = replacement
            return True

    def list_rows(
        self,
        partition_key: str,
        *,
        row_key_prefix: str,
        limit: int,
        cursor: str | None,
        sort_path: str,
    ) -> tuple[tuple[_PartitionRow, ...], str | None]:
        del sort_path
        with self._lock:
            rows = [
                row
                for (partition, _), row in self._rows.items()
                if partition == partition_key and row.row_key.startswith(row_key_prefix)
            ]
        rows.sort(
            key=lambda row: decode_execution_lineage_admission_record(row.data).admission_position,
        )
        start = 0
        if cursor is not None:
            for index, row in enumerate(rows):
                if row.row_key == cursor:
                    start = index + 1
                    break
        page = tuple(rows[start : start + limit])
        next_cursor = page[-1].row_key if len(rows) > start + limit else None
        return page, next_cursor


class InMemoryExecutionLineagePersistence(ExecutionLineagePersistence):
    """Thread-safe in-memory lineage persistence for tests."""

    def __init__(self) -> None:
        self._logic = _ExecutionLineageStoreLogic(_InMemoryPartitionRowStore())

    @property
    def is_durable(self) -> bool:
        return False

    def open_attempt(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageAttemptState:
        return self._logic.open_attempt(scope)

    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.open_segment(scope, root_execution_id, predecessor_root_execution_id)

    def admit_root(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        return self._logic.admit_root(
            scope,
            segment_root_execution_id,
            execution_id,
            graph_node_id=graph_node_id,
        )

    def admit_child(
        self,
        scope: ExecutionLineageAttemptScope,
        segment_root_execution_id: ExecutionId,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        *,
        graph_node_id: str | None = None,
    ) -> ExecutionLineageAdmissionRecord:
        return self._logic.admit_child(
            scope,
            segment_root_execution_id,
            execution_id,
            parent_execution_id,
            graph_node_id=graph_node_id,
        )

    def close_segment_for_resume(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.close_segment_for_resume(scope, root_execution_id)

    def mark_degraded(
        self,
        scope: ExecutionLineageAttemptScope,
        reason_code: str,
    ) -> ExecutionLineageAttemptState:
        return self._logic.mark_degraded(scope, reason_code)

    def seal_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        closure_kind: ExecutionLineageAttemptClosureKind,
    ) -> ExecutionLineageSealRecord:
        return self._logic.seal_attempt(scope, closure_kind)

    def list_admissions_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAdmissionPage:
        return self._logic.list_admissions_for_attempt(scope, limit, cursor=cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._logic.read_attempt_lineage_state(scope)

    def read_seal(self, scope: ExecutionLineageAttemptScope) -> ExecutionLineageSealRecord | None:
        return self._logic.read_seal(scope)
