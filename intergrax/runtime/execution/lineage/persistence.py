# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Execution lineage persistence implementations (DG-001 R1)."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    validate_attempt_id,
    validate_execution_id,
)
from intergrax.contracts.execution_lineage import (
    ExecutionLineageAdmissionPage,
    ExecutionLineageAdmissionRecord,
    ExecutionLineageAttemptClosureKind,
    ExecutionLineageAttemptDiscoveryPage,
    ExecutionLineageAttemptDiscoveryRecord,
    ExecutionLineageAttemptScope,
    ExecutionLineageAttemptState,
    ExecutionLineageConfigurationError,
    ExecutionLineageDiscoveryRunState,
    ExecutionLineageError,
    ExecutionLineageIntegrityError,
    ExecutionLineagePersistence,
    ExecutionLineageRunScope,
    ExecutionLineageSealRecord,
    ExecutionLineageSegmentLifecycle,
    ExecutionLineageSegmentPage,
    ExecutionLineageSegmentRecord,
    ExecutionLineageUnavailableError,
    build_execution_lineage_run_scope,
    validate_admission_page_limit,
    validate_lineage_page_limit,
)
from intergrax.runtime.execution.lineage.codecs import (
    decode_execution_lineage_admission_record,
    decode_execution_lineage_attempt_discovery_record,
    decode_execution_lineage_attempt_state,
    decode_execution_lineage_discovery_run_state,
    decode_execution_lineage_segment_record,
    decode_execution_lineage_seal_record,
    encode_execution_lineage_admission_record,
    encode_execution_lineage_attempt_discovery_record,
    encode_execution_lineage_attempt_state,
    encode_execution_lineage_discovery_run_state,
    encode_execution_lineage_segment_record,
    encode_execution_lineage_seal_record,
)

_PARTITION_PREFIX = "intergrax.execution_lineage.v1"
_META_ROW = "meta:attempt"
_DISCOVERY_RUN_META_ROW = "meta:discovery_run"
_DISCOVERY_ATTEMPT_ROW_PREFIX = "attempt:"
_SEAL_ROW = "meta:seal"
_SEGMENT_ROW_PREFIX = "segment:"
_ADMISSION_ROW_PREFIX = "admission:"
_MAX_ATOMIC_RETRIES = 256
_MAX_ON_CREATED_OPS = 4
_LIST_ROW_SORT_EXTRACTORS: dict[str, Callable[[dict[str, object]], object]] = {}
_DecodedPayload = TypeVar("_DecodedPayload")


def execution_lineage_partition_key(scope: ExecutionLineageAttemptScope) -> str:
    return (
        f"{_PARTITION_PREFIX}:tenant:{scope.tenant_id}:task:{scope.task_id}:"
        f"run:{scope.run_id}:attempt:{scope.attempt_id}"
    )


def execution_lineage_discovery_partition_key(
    run_scope: ExecutionLineageRunScope,
) -> str:
    return (
        f"{_PARTITION_PREFIX}:tenant:{run_scope.tenant_id}:task:{run_scope.task_id}:"
        f"run:{run_scope.run_id}:discovery"
    )


def _discovery_attempt_row_key(attempt_id: AttemptId) -> str:
    return f"{_DISCOVERY_ATTEMPT_ROW_PREFIX}{attempt_id}"


def _register_list_row_sort_extractor(
    sort_path: str,
    extractor: Callable[[dict[str, object]], object],
) -> None:
    _LIST_ROW_SORT_EXTRACTORS[sort_path] = extractor


def _segment_row_key(root_execution_id: ExecutionId) -> str:
    return f"{_SEGMENT_ROW_PREFIX}{root_execution_id}"


def _admission_row_key(execution_id: ExecutionId) -> str:
    return f"{_ADMISSION_ROW_PREFIX}{execution_id}"


def _scopes_match(
    left: ExecutionLineageAttemptScope, right: ExecutionLineageAttemptScope
) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.task_id == right.task_id
        and left.run_id == right.run_id
        and left.attempt_id == right.attempt_id
    )


def _run_scopes_match(
    left: ExecutionLineageRunScope, right: ExecutionLineageRunScope
) -> bool:
    return (
        left.tenant_id == right.tenant_id
        and left.task_id == right.task_id
        and left.run_id == right.run_id
    )


def _initial_attempt_state(
    scope: ExecutionLineageAttemptScope,
    *,
    discovery_contract_version: int | None = None,
) -> ExecutionLineageAttemptState:
    return ExecutionLineageAttemptState(
        scope=scope,
        generation=1,
        next_admission_position=1,
        active_segment_root_execution_id=None,
        degraded=False,
        sealed=False,
        closure_kind=None,
        discovery_contract_version=discovery_contract_version,
    )


def _initial_discovery_run_state(
    run_scope: ExecutionLineageRunScope,
) -> ExecutionLineageDiscoveryRunState:
    return ExecutionLineageDiscoveryRunState(
        run_scope=run_scope,
        generation=1,
        next_discovery_position=2,
        coverage_contract_version=None,
        coverage_origin=None,
    )


@dataclass(frozen=True, slots=True)
class _PartitionRow:
    partition_key: str
    row_key: str
    data: dict[str, object]


def _extract_list_row_sort_value(
    row: _PartitionRow,
    *,
    sort_path: str,
) -> object:
    if sort_path == "root_execution_id":
        return row.row_key
    extractor = _LIST_ROW_SORT_EXTRACTORS.get(sort_path)
    if extractor is None:
        raise ValueError(f"unsupported list row sort path: {sort_path}")
    return extractor(row.data)


@dataclass(frozen=True, slots=True)
class _PartitionPutIfAbsentOnCreated:
    row: _PartitionRow


@dataclass(frozen=True, slots=True)
class _PartitionReplaceIfMatchOnCreated:
    expected: _PartitionRow
    replacement: _PartitionRow


_PartitionOnCreatedRowOp = (
    _PartitionPutIfAbsentOnCreated | _PartitionReplaceIfMatchOnCreated
)


@dataclass(frozen=True, slots=True)
class _PartitionAtomicRowBatch:
    partition_key: str
    primary_put_if_absent: _PartitionRow
    on_created_ops: tuple[_PartitionOnCreatedRowOp, ...] = ()


@dataclass(frozen=True, slots=True)
class _PartitionAtomicRowBatchResult:
    primary_created: bool


def _validate_partition_atomic_row_batch(
    batch: _PartitionAtomicRowBatch,
) -> _PartitionAtomicRowBatch:
    if not batch.partition_key:
        raise ValueError("partition_atomic_row_batch_partition_key_invalid")
    primary = batch.primary_put_if_absent
    if primary.partition_key != batch.partition_key:
        raise ValueError("partition_atomic_row_batch_primary_partition_mismatch")
    if len(batch.on_created_ops) > _MAX_ON_CREATED_OPS:
        raise ValueError("partition_atomic_row_batch_on_created_ops_exceeded")
    for op in batch.on_created_ops:
        if isinstance(op, _PartitionPutIfAbsentOnCreated):
            if op.row.partition_key != batch.partition_key:
                raise ValueError(
                    "partition_atomic_row_batch_on_created_partition_mismatch"
                )
        elif isinstance(op, _PartitionReplaceIfMatchOnCreated):
            if op.expected.partition_key != batch.partition_key:
                raise ValueError(
                    "partition_atomic_row_batch_on_created_partition_mismatch"
                )
            if op.replacement.partition_key != batch.partition_key:
                raise ValueError(
                    "partition_atomic_row_batch_on_created_partition_mismatch"
                )
            if op.expected.row_key != op.replacement.row_key:
                raise ValueError(
                    "partition_atomic_row_batch_on_created_row_key_mismatch"
                )
        else:
            raise TypeError("partition_atomic_row_batch_on_created_op_invalid")
    return batch


class _PartitionAtomicRowStore(Protocol):
    def get_row(self, partition_key: str, row_key: str) -> _PartitionRow | None: ...

    def put_if_absent(self, row: _PartitionRow) -> bool: ...

    def replace_if_match(
        self, expected: _PartitionRow, replacement: _PartitionRow
    ) -> bool: ...

    def list_rows(
        self,
        partition_key: str,
        *,
        row_key_prefix: str,
        limit: int,
        cursor: str | None,
        sort_path: str,
    ) -> tuple[tuple[_PartitionRow, ...], str | None]: ...

    def execute_partition_atomic_batch(
        self,
        batch: _PartitionAtomicRowBatch,
    ) -> _PartitionAtomicRowBatchResult: ...


class _ExecutionLineageStoreLogic:
    """Shared lineage semantics over a partition-atomic row store."""

    def __init__(self, store: _PartitionAtomicRowStore) -> None:
        self._store = store

    def open_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        *,
        discovery_contract_version: int | None = None,
    ) -> ExecutionLineageAttemptState:
        if discovery_contract_version is not None and discovery_contract_version != 1:
            raise ExecutionLineageIntegrityError("invalid discovery_contract_version")
        partition = execution_lineage_partition_key(scope)
        existing = self._read_attempt_state(partition)
        if existing is not None:
            return self._open_existing_attempt(
                scope,
                existing,
                requested_discovery_contract_version=discovery_contract_version,
            )
        if discovery_contract_version is None:
            raise ExecutionLineageConfigurationError(
                "new attempt requires discovery_contract_version=1",
            )
        run_scope = build_execution_lineage_run_scope(
            tenant_id=scope.tenant_id,
            task_id=scope.task_id,
            run_id=scope.run_id,
        )
        discovery = self.read_attempt_discovery_record(run_scope, scope.attempt_id)
        if discovery is None:
            raise ExecutionLineageIntegrityError(
                "post-v1 attempt missing discovery record",
            )
        initial = _initial_attempt_state(scope, discovery_contract_version=1)
        created = self._store.put_if_absent(
            _PartitionRow(
                partition, _META_ROW, encode_execution_lineage_attempt_state(initial)
            ),
        )
        if created:
            return initial
        loaded = self._read_attempt_state(partition)
        if loaded is None:
            raise ExecutionLineageUnavailableError(
                "attempt open race left no durable state",
            )
        return self._open_existing_attempt(
            scope,
            loaded,
            requested_discovery_contract_version=discovery_contract_version,
        )

    def _open_existing_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        existing: ExecutionLineageAttemptState,
        *,
        requested_discovery_contract_version: int | None,
    ) -> ExecutionLineageAttemptState:
        if not _scopes_match(existing.scope, scope):
            raise ExecutionLineageIntegrityError("attempt scope mismatch")
        existing_marker = existing.discovery_contract_version
        requested = requested_discovery_contract_version
        if existing_marker is None and requested is None:
            return existing
        if existing_marker == 1 and requested == 1:
            run_scope = build_execution_lineage_run_scope(
                tenant_id=scope.tenant_id,
                task_id=scope.task_id,
                run_id=scope.run_id,
            )
            if self.read_attempt_discovery_record(run_scope, scope.attempt_id) is None:
                raise ExecutionLineageIntegrityError(
                    "post-v1 attempt missing discovery record",
                )
            return existing
        if existing_marker is None and requested == 1:
            raise ExecutionLineageConfigurationError(
                "legacy attempt cannot be promoted to discovery-v1",
            )
        if existing_marker == 1 and requested is None:
            raise ExecutionLineageIntegrityError(
                "discovery-v1 attempt requires discovery_contract_version=1",
            )
        if existing_marker != requested:
            raise ExecutionLineageIntegrityError("discovery contract mismatch")
        return existing

    def register_attempt_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord:
        attempt = validate_attempt_id(attempt_id)
        partition = execution_lineage_discovery_partition_key(run_scope)
        row_key = _discovery_attempt_row_key(attempt)
        existing = self._read_attempt_discovery_record(partition, row_key)
        if existing is not None:
            if existing.attempt_id != attempt:
                raise ExecutionLineageIntegrityError("discovery attempt_id mismatch")
            if not _run_scopes_match(existing.run_scope, run_scope):
                raise ExecutionLineageIntegrityError("discovery run scope mismatch")
            return existing
        for _ in range(_MAX_ATOMIC_RETRIES):
            run_state = self._read_discovery_run_state(partition)
            if run_state is None:
                position = 1
                record = ExecutionLineageAttemptDiscoveryRecord(
                    run_scope=run_scope,
                    attempt_id=attempt,
                    discovery_position=position,
                )
                initial_run_state = _initial_discovery_run_state(run_scope)
                batch = _PartitionAtomicRowBatch(
                    partition_key=partition,
                    primary_put_if_absent=_PartitionRow(
                        partition,
                        row_key,
                        encode_execution_lineage_attempt_discovery_record(record),
                    ),
                    on_created_ops=(
                        _PartitionPutIfAbsentOnCreated(
                            row=_PartitionRow(
                                partition,
                                _DISCOVERY_RUN_META_ROW,
                                encode_execution_lineage_discovery_run_state(
                                    initial_run_state,
                                ),
                            ),
                        ),
                    ),
                )
                if self._execute_discovery_registration_batch(batch):
                    return record
                continue
            position = run_state.next_discovery_position
            record = ExecutionLineageAttemptDiscoveryRecord(
                run_scope=run_scope,
                attempt_id=attempt,
                discovery_position=position,
            )
            updated_run_state = run_state.model_copy(
                update={
                    "generation": run_state.generation + 1,
                    "next_discovery_position": position + 1,
                },
            )
            batch = _PartitionAtomicRowBatch(
                partition_key=partition,
                primary_put_if_absent=_PartitionRow(
                    partition,
                    row_key,
                    encode_execution_lineage_attempt_discovery_record(record),
                ),
                on_created_ops=(
                    _PartitionReplaceIfMatchOnCreated(
                        expected=self._discovery_run_row(partition, run_state),
                        replacement=self._discovery_run_row(
                            partition,
                            updated_run_state,
                        ),
                    ),
                ),
            )
            if self._execute_discovery_registration_batch(batch):
                return record
        raise ExecutionLineageUnavailableError(
            "failed to register attempt discovery after bounded retries",
        )

    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        partition = execution_lineage_discovery_partition_key(run_scope)
        return self._read_discovery_run_state(partition)

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        validated_limit = validate_lineage_page_limit(limit)
        partition = execution_lineage_discovery_partition_key(run_scope)
        rows, next_cursor = self._store.list_rows(
            partition,
            row_key_prefix=_DISCOVERY_ATTEMPT_ROW_PREFIX,
            limit=validated_limit,
            cursor=cursor,
            sort_path="discovery_position",
        )
        attempts = tuple(
            self._decode_attempt_discovery_record(row.data) for row in rows
        )
        for record in attempts:
            if not _run_scopes_match(record.run_scope, run_scope):
                raise ExecutionLineageIntegrityError("discovery run scope mismatch")
        return ExecutionLineageAttemptDiscoveryPage(
            attempts=attempts,
            next_cursor=next_cursor,
        )

    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        partition = execution_lineage_discovery_partition_key(run_scope)
        row_key = _discovery_attempt_row_key(validate_attempt_id(attempt_id))
        return self._read_attempt_discovery_record(partition, row_key)

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
            raise ExecutionLineageIntegrityError(
                "segment predecessor cannot equal current root"
            )
        partition = execution_lineage_partition_key(scope)
        for _ in range(_MAX_ATOMIC_RETRIES):
            attempt_state = self._require_attempt_state(partition, scope)
            if attempt_state.sealed:
                raise ExecutionLineageIntegrityError(
                    "cannot open segment on sealed attempt"
                )
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
            predecessor_unclean: (
                tuple[
                    ExecutionLineageSegmentRecord,
                    ExecutionLineageSegmentRecord,
                ]
                | None
            ) = None
            if predecessor is not None:
                predecessor_segment = self._read_segment(
                    partition, _segment_row_key(predecessor)
                )
                if predecessor_segment is None:
                    raise ExecutionLineageIntegrityError("predecessor segment missing")
                if (
                    predecessor_segment.lifecycle
                    is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN
                ):
                    predecessor_unclean = (
                        predecessor_segment,
                        predecessor_segment.model_copy(
                            update={
                                "lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_UNCLEAN
                            },
                        ),
                    )
            if self._execute_segment_open_batch(
                partition,
                attempt_state,
                updated_attempt,
                segment_key,
                new_segment,
                predecessor_unclean=predecessor_unclean,
            ):
                return new_segment
        raise ExecutionLineageUnavailableError(
            "failed to open segment after bounded retries"
        )

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
                raise ExecutionLineageIntegrityError(
                    "cannot close segment on sealed attempt"
                )
            segment = self._read_segment(partition, _segment_row_key(root))
            if segment is None:
                raise ExecutionLineageIntegrityError(
                    "segment not found for clean close"
                )
            if (
                segment.lifecycle
                is ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN
            ):
                return segment
            if segment.lifecycle is not ExecutionLineageSegmentLifecycle.SEGMENT_OPEN:
                raise ExecutionLineageIntegrityError("segment not open for clean close")
            closed = segment.model_copy(
                update={
                    "lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN
                },
            )
            if self._replace_segment(partition, segment, closed):
                return closed
        raise ExecutionLineageUnavailableError(
            "failed to close segment after bounded retries"
        )

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
            updated = attempt_state.model_copy(
                update={"generation": attempt_state.generation + 1, "degraded": True},
            )
            if self._replace_attempt_state(partition, attempt_state, updated):
                return updated
        raise ExecutionLineageUnavailableError(
            "failed to mark degraded after bounded retries"
        )

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
                    raise ExecutionLineageIntegrityError(
                        "sealed attempt without seal record"
                    )
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
            segment_close: (
                tuple[ExecutionLineageSegmentRecord, ExecutionLineageSegmentRecord]
                | None
            ) = None
            if active_root is not None:
                segment = self._read_segment(partition, _segment_row_key(active_root))
                if (
                    segment is not None
                    and segment.lifecycle
                    is ExecutionLineageSegmentLifecycle.SEGMENT_OPEN
                ):
                    segment_close = (
                        segment,
                        segment.model_copy(
                            update={
                                "lifecycle": ExecutionLineageSegmentLifecycle.SEGMENT_CLOSED_CLEAN
                            },
                        ),
                    )
            if self._execute_seal_batch(
                partition,
                attempt_state,
                updated_attempt,
                seal,
                segment_close=segment_close,
            ):
                return seal
        raise ExecutionLineageUnavailableError(
            "failed to seal attempt after bounded retries"
        )

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
        admissions = tuple(self._decode_admission_record(row.data) for row in rows)
        admissions = tuple(
            sorted(admissions, key=lambda item: item.admission_position),
        )
        return ExecutionLineageAdmissionPage(
            admissions=admissions, next_cursor=next_cursor
        )

    def list_segments_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageSegmentPage:
        validated_limit = validate_lineage_page_limit(limit)
        partition = execution_lineage_partition_key(scope)
        rows, next_cursor = self._store.list_rows(
            partition,
            row_key_prefix=_SEGMENT_ROW_PREFIX,
            limit=validated_limit,
            cursor=cursor,
            sort_path="root_execution_id",
        )
        segments = tuple(self._decode_segment_record(row.data) for row in rows)
        segments = tuple(
            sorted(segments, key=lambda item: str(item.root_execution_id)),
        )
        return ExecutionLineageSegmentPage(segments=segments, next_cursor=next_cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._read_attempt_state(execution_lineage_partition_key(scope))

    def read_seal(
        self, scope: ExecutionLineageAttemptScope
    ) -> ExecutionLineageSealRecord | None:
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
            if (
                attempt_state.active_segment_root_execution_id
                != segment_root_execution_id
            ):
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
            if self._execute_admission_batch(
                partition,
                attempt_state,
                updated_attempt,
                admission_key,
                record,
            ):
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
        predecessor_segment = self._read_segment(
            partition, _segment_row_key(predecessor)
        )
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

    def _attempt_row(
        self,
        partition: str,
        state: ExecutionLineageAttemptState,
    ) -> _PartitionRow:
        return _PartitionRow(
            partition, _META_ROW, encode_execution_lineage_attempt_state(state)
        )

    def _segment_row(
        self,
        partition: str,
        segment: ExecutionLineageSegmentRecord,
    ) -> _PartitionRow:
        return _PartitionRow(
            partition,
            _segment_row_key(segment.root_execution_id),
            encode_execution_lineage_segment_record(segment),
        )

    def _execute_admission_batch(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        admission_row_key: str,
        record: ExecutionLineageAdmissionRecord,
    ) -> bool:
        batch = _PartitionAtomicRowBatch(
            partition_key=partition,
            primary_put_if_absent=_PartitionRow(
                partition,
                admission_row_key,
                encode_execution_lineage_admission_record(record),
            ),
            on_created_ops=(
                _PartitionReplaceIfMatchOnCreated(
                    expected=self._attempt_row(partition, attempt_expected),
                    replacement=self._attempt_row(partition, attempt_replacement),
                ),
            ),
        )
        try:
            return self._store.execute_partition_atomic_batch(batch).primary_created
        except RuntimeError:
            return False

    def _execute_segment_open_batch(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        segment_row_key: str,
        segment: ExecutionLineageSegmentRecord,
        *,
        predecessor_unclean: tuple[
            ExecutionLineageSegmentRecord,
            ExecutionLineageSegmentRecord,
        ]
        | None = None,
    ) -> bool:
        on_created: list[_PartitionOnCreatedRowOp] = []
        if predecessor_unclean is not None:
            predecessor_expected, predecessor_replacement = predecessor_unclean
            on_created.append(
                _PartitionReplaceIfMatchOnCreated(
                    expected=self._segment_row(partition, predecessor_expected),
                    replacement=self._segment_row(partition, predecessor_replacement),
                ),
            )
        on_created.append(
            _PartitionReplaceIfMatchOnCreated(
                expected=self._attempt_row(partition, attempt_expected),
                replacement=self._attempt_row(partition, attempt_replacement),
            ),
        )
        batch = _PartitionAtomicRowBatch(
            partition_key=partition,
            primary_put_if_absent=_PartitionRow(
                partition,
                segment_row_key,
                encode_execution_lineage_segment_record(segment),
            ),
            on_created_ops=tuple(on_created),
        )
        try:
            return self._store.execute_partition_atomic_batch(batch).primary_created
        except RuntimeError:
            return False

    def _execute_seal_batch(
        self,
        partition: str,
        attempt_expected: ExecutionLineageAttemptState,
        attempt_replacement: ExecutionLineageAttemptState,
        seal: ExecutionLineageSealRecord,
        *,
        segment_close: tuple[
            ExecutionLineageSegmentRecord, ExecutionLineageSegmentRecord
        ]
        | None,
    ) -> bool:
        on_created: list[_PartitionOnCreatedRowOp] = []
        if segment_close is not None:
            segment_expected, segment_replacement = segment_close
            on_created.append(
                _PartitionReplaceIfMatchOnCreated(
                    expected=self._segment_row(partition, segment_expected),
                    replacement=self._segment_row(partition, segment_replacement),
                ),
            )
        on_created.append(
            _PartitionReplaceIfMatchOnCreated(
                expected=self._attempt_row(partition, attempt_expected),
                replacement=self._attempt_row(partition, attempt_replacement),
            ),
        )
        batch = _PartitionAtomicRowBatch(
            partition_key=partition,
            primary_put_if_absent=_PartitionRow(
                partition,
                _SEAL_ROW,
                encode_execution_lineage_seal_record(seal),
            ),
            on_created_ops=tuple(on_created),
        )
        try:
            return self._store.execute_partition_atomic_batch(batch).primary_created
        except RuntimeError:
            return False

    def _execute_discovery_registration_batch(
        self,
        batch: _PartitionAtomicRowBatch,
    ) -> bool:
        try:
            return self._store.execute_partition_atomic_batch(batch).primary_created
        except RuntimeError:
            return False

    def _discovery_run_row(
        self,
        partition: str,
        state: ExecutionLineageDiscoveryRunState,
    ) -> _PartitionRow:
        return _PartitionRow(
            partition,
            _DISCOVERY_RUN_META_ROW,
            encode_execution_lineage_discovery_run_state(state),
        )

    def _decode_durable_payload(
        self,
        payload: dict[str, object],
        *,
        decode: Callable[[dict[str, object]], _DecodedPayload],
    ) -> _DecodedPayload:
        try:
            return decode(payload)
        except ExecutionLineageIntegrityError:
            raise
        except (ExecutionLineageError, ValueError, KeyError, TypeError) as exc:
            raise ExecutionLineageIntegrityError(str(exc)) from exc

    def _decode_attempt_state(
        self, payload: dict[str, object]
    ) -> ExecutionLineageAttemptState:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_attempt_state,
        )

    def _decode_segment_record(
        self, payload: dict[str, object]
    ) -> ExecutionLineageSegmentRecord:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_segment_record,
        )

    def _decode_admission_record(
        self, payload: dict[str, object]
    ) -> ExecutionLineageAdmissionRecord:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_admission_record,
        )

    def _decode_seal_record(
        self, payload: dict[str, object]
    ) -> ExecutionLineageSealRecord:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_seal_record,
        )

    def _decode_attempt_discovery_record(
        self, payload: dict[str, object]
    ) -> ExecutionLineageAttemptDiscoveryRecord:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_attempt_discovery_record,
        )

    def _decode_discovery_run_state(
        self, payload: dict[str, object]
    ) -> ExecutionLineageDiscoveryRunState:
        return self._decode_durable_payload(
            payload,
            decode=decode_execution_lineage_discovery_run_state,
        )

    def _read_attempt_state(
        self, partition: str
    ) -> ExecutionLineageAttemptState | None:
        row = self._store.get_row(partition, _META_ROW)
        if row is None:
            return None
        return self._decode_attempt_state(row.data)

    def _read_discovery_run_state(
        self, partition: str
    ) -> ExecutionLineageDiscoveryRunState | None:
        row = self._store.get_row(partition, _DISCOVERY_RUN_META_ROW)
        if row is None:
            return None
        return self._decode_discovery_run_state(row.data)

    def _read_attempt_discovery_record(
        self, partition: str, row_key: str
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        row = self._store.get_row(partition, row_key)
        if row is None:
            return None
        return self._decode_attempt_discovery_record(row.data)

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

    def _read_segment(
        self, partition: str, row_key: str
    ) -> ExecutionLineageSegmentRecord | None:
        row = self._store.get_row(partition, row_key)
        if row is None:
            return None
        return self._decode_segment_record(row.data)

    def _read_admission(
        self, partition: str, row_key: str
    ) -> ExecutionLineageAdmissionRecord | None:
        row = self._store.get_row(partition, row_key)
        if row is None:
            return None
        return self._decode_admission_record(row.data)

    def _read_seal(self, partition: str) -> ExecutionLineageSealRecord | None:
        row = self._store.get_row(partition, _SEAL_ROW)
        if row is None:
            return None
        return self._decode_seal_record(row.data)

    def _replace_attempt_state(
        self,
        partition: str,
        expected: ExecutionLineageAttemptState,
        replacement: ExecutionLineageAttemptState,
    ) -> bool:
        return self._store.replace_if_match(
            self._attempt_row(partition, expected),
            self._attempt_row(partition, replacement),
        )

    def _replace_segment(
        self,
        partition: str,
        expected: ExecutionLineageSegmentRecord,
        replacement: ExecutionLineageSegmentRecord,
    ) -> bool:
        return self._store.replace_if_match(
            self._segment_row(partition, expected),
            self._segment_row(partition, replacement),
        )


class _InMemoryPartitionAtomicRowStore:
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

    def replace_if_match(
        self, expected: _PartitionRow, replacement: _PartitionRow
    ) -> bool:
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
        with self._lock:
            rows = [
                row
                for (partition, _), row in self._rows.items()
                if partition == partition_key and row.row_key.startswith(row_key_prefix)
            ]
        rows.sort(
            key=lambda row: (
                _extract_list_row_sort_value(row, sort_path=sort_path),
                row.row_key,
            )
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

    def execute_partition_atomic_batch(
        self,
        batch: _PartitionAtomicRowBatch,
    ) -> _PartitionAtomicRowBatchResult:
        validated = _validate_partition_atomic_row_batch(batch)
        with self._lock:
            snapshot = dict(self._rows)
            primary_created = self._put_if_absent_unlocked(
                validated.primary_put_if_absent,
                rows=snapshot,
            )
            if primary_created:
                for op in validated.on_created_ops:
                    if isinstance(op, _PartitionPutIfAbsentOnCreated):
                        if not self._put_if_absent_unlocked(op.row, rows=snapshot):
                            raise RuntimeError(
                                "partition_atomic_row_batch_on_created_conflict"
                            )
                    elif isinstance(op, _PartitionReplaceIfMatchOnCreated):
                        if not self._replace_if_match_unlocked(
                            expected=op.expected,
                            replacement=op.replacement,
                            rows=snapshot,
                        ):
                            raise RuntimeError(
                                "partition_atomic_row_batch_on_created_stale"
                            )
                    else:
                        raise TypeError(
                            "partition_atomic_row_batch_on_created_op_invalid"
                        )
            self._rows = snapshot
            return _PartitionAtomicRowBatchResult(primary_created=primary_created)

    @staticmethod
    def _put_if_absent_unlocked(
        row: _PartitionRow,
        *,
        rows: dict[tuple[str, str], _PartitionRow],
    ) -> bool:
        key = (row.partition_key, row.row_key)
        if key in rows:
            return False
        rows[key] = row
        return True

    @staticmethod
    def _replace_if_match_unlocked(
        *,
        expected: _PartitionRow,
        replacement: _PartitionRow,
        rows: dict[tuple[str, str], _PartitionRow],
    ) -> bool:
        current = rows.get((expected.partition_key, expected.row_key))
        if current is None or current.data != expected.data:
            return False
        rows[(replacement.partition_key, replacement.row_key)] = replacement
        return True


class InMemoryExecutionLineagePersistence(ExecutionLineagePersistence):
    """Thread-safe in-memory lineage persistence for tests."""

    def __init__(self) -> None:
        self._logic = _ExecutionLineageStoreLogic(_InMemoryPartitionAtomicRowStore())

    @property
    def is_durable(self) -> bool:
        return False

    def open_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        *,
        discovery_contract_version: int | None = None,
    ) -> ExecutionLineageAttemptState:
        return self._logic.open_attempt(
            scope,
            discovery_contract_version=discovery_contract_version,
        )

    def register_attempt_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord:
        return self._logic.register_attempt_for_run(run_scope, attempt_id)

    def open_segment(
        self,
        scope: ExecutionLineageAttemptScope,
        root_execution_id: ExecutionId,
        predecessor_root_execution_id: ExecutionId | None = None,
    ) -> ExecutionLineageSegmentRecord:
        return self._logic.open_segment(
            scope, root_execution_id, predecessor_root_execution_id
        )

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

    def list_segments_for_attempt(
        self,
        scope: ExecutionLineageAttemptScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageSegmentPage:
        return self._logic.list_segments_for_attempt(scope, limit, cursor=cursor)

    def read_attempt_lineage_state(
        self,
        scope: ExecutionLineageAttemptScope,
    ) -> ExecutionLineageAttemptState | None:
        return self._logic.read_attempt_lineage_state(scope)

    def read_seal(
        self, scope: ExecutionLineageAttemptScope
    ) -> ExecutionLineageSealRecord | None:
        return self._logic.read_seal(scope)

    def read_discovery_run_state(
        self,
        run_scope: ExecutionLineageRunScope,
    ) -> ExecutionLineageDiscoveryRunState | None:
        return self._logic.read_discovery_run_state(run_scope)

    def list_attempts_for_run(
        self,
        run_scope: ExecutionLineageRunScope,
        limit: int,
        cursor: str | None = None,
    ) -> ExecutionLineageAttemptDiscoveryPage:
        return self._logic.list_attempts_for_run(run_scope, limit, cursor=cursor)

    def read_attempt_discovery_record(
        self,
        run_scope: ExecutionLineageRunScope,
        attempt_id: AttemptId,
    ) -> ExecutionLineageAttemptDiscoveryRecord | None:
        return self._logic.read_attempt_discovery_record(run_scope, attempt_id)


_register_list_row_sort_extractor(
    "admission_position",
    lambda payload: (
        decode_execution_lineage_admission_record(payload).admission_position
    ),
)
_register_list_row_sort_extractor(
    "discovery_position",
    lambda payload: (
        decode_execution_lineage_attempt_discovery_record(
            payload,
        ).discovery_position
    ),
)
