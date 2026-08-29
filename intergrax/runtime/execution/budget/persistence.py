# © Artur Czarnecki. All rights reserved.

"""Durable per-Run execution budget persistence (UE-9AR1)."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    validate_attempt_id,
    validate_execution_id,
    validate_run_id,
)
from intergrax.distributed.contracts.kv_store import DistributedKVStore
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.runtime.execution.budget.ledger import (
    ExecutionBudgetLedger,
    ExecutionBudgetLedgerFactory,
    InMemoryExecutionBudgetLedger,
    create_execution_budget_ledger,
)
from intergrax.runtime.execution.budget.models import (
    BudgetReservationScope,
    BudgetUsageTotals,
    ChildBudgetAllocationDecision,
    ExecutionBudgetAllocationMode,
    ExecutionBudgetReservationGrant,
)
from intergrax.runtime.execution.budget.snapshot import (
    PersistedBudgetRecord,
    RunBudgetLedgerSnapshot,
    RunBudgetLedgerSnapshotError,
    validate_snapshot_attempt_id,
    validate_snapshot_schema_version,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_KV_KEY_PREFIX = "run_budget_ledger"
_DOCUMENT_STORE_PARTITION_PREFIX = "intergrax.run_budget_ledger.v1"
_SNAPSHOT_SCHEMA_VERSION = 1


class RunBudgetPersistenceError(RuntimeError):
    """Raised when durable budget state cannot be loaded or stored safely."""


class RunBudgetPersistence(ABC):
    """Platform-owned durable per-Run budget ledger snapshots."""

    @abstractmethod
    def load_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
    ) -> bytes | None:
        """Return encoded snapshot bytes or ``None`` when absent."""

    @abstractmethod
    def compare_and_swap_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        snapshot: RunBudgetLedgerSnapshot,
    ) -> bool:
        """Atomically store ``snapshot`` when ``expected`` matches current bytes."""


def _kv_storage_key(run_id: RunId) -> str:
    return f"{_KV_KEY_PREFIX}:{run_id}"


def _document_partition(tenant_id: str) -> str:
    return f"{_DOCUMENT_STORE_PARTITION_PREFIX}:{tenant_id}"


def _document_row_key(run_id: RunId) -> str:
    return str(run_id)


def _usage_totals_to_json(totals: BudgetUsageTotals) -> dict[str, int | float]:
    return {
        "input_tokens": totals.input_tokens,
        "output_tokens": totals.output_tokens,
        "total_tokens": totals.total_tokens,
        "llm_calls": totals.llm_calls,
        "tool_calls": totals.tool_calls,
        "rag_invocations": totals.rag_invocations,
        "websearch_invocations": totals.websearch_invocations,
        "wall_time_seconds": totals.wall_time_seconds,
        "planner_iterations": totals.planner_iterations,
        "replans": totals.replans,
    }


def _usage_totals_from_json(raw: object) -> BudgetUsageTotals:
    if not isinstance(raw, dict):
        raise RunBudgetLedgerSnapshotError("budget usage totals must be an object")
    return BudgetUsageTotals(
        input_tokens=_require_int(raw.get("input_tokens"), field="input_tokens"),
        output_tokens=_require_int(raw.get("output_tokens"), field="output_tokens"),
        total_tokens=_require_int(raw.get("total_tokens"), field="total_tokens"),
        llm_calls=_require_int(raw.get("llm_calls"), field="llm_calls"),
        tool_calls=_require_int(raw.get("tool_calls"), field="tool_calls"),
        rag_invocations=_require_int(raw.get("rag_invocations"), field="rag_invocations"),
        websearch_invocations=_require_int(
            raw.get("websearch_invocations"),
            field="websearch_invocations",
        ),
        wall_time_seconds=_require_float(raw.get("wall_time_seconds"), field="wall_time_seconds"),
        planner_iterations=_require_int(raw.get("planner_iterations"), field="planner_iterations"),
        replans=_require_int(raw.get("replans"), field="replans"),
    )


def _optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    if not isinstance(raw, int):
        raise RunBudgetLedgerSnapshotError("run budget limit must be int or null")
    return raw


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    if not isinstance(raw, (int, float)):
        raise RunBudgetLedgerSnapshotError("run budget limit must be number or null")
    return float(raw)


def _run_budget_to_json(budget: RunBudget) -> dict[str, int | float | None]:
    return {
        "max_input_tokens": budget.max_input_tokens,
        "max_output_tokens": budget.max_output_tokens,
        "max_total_tokens": budget.max_total_tokens,
        "max_llm_calls": budget.max_llm_calls,
        "max_tool_calls": budget.max_tool_calls,
        "max_rag_invocations": budget.max_rag_invocations,
        "max_websearch_invocations": budget.max_websearch_invocations,
        "max_wall_time_seconds": budget.max_wall_time_seconds,
        "max_planner_iterations": budget.max_planner_iterations,
        "max_replans": budget.max_replans,
    }


def _run_budget_from_json(raw: object) -> RunBudget:
    if not isinstance(raw, dict):
        raise RunBudgetLedgerSnapshotError("run budget limits must be an object")
    return RunBudget(
        max_input_tokens=_optional_int(raw.get("max_input_tokens")),
        max_output_tokens=_optional_int(raw.get("max_output_tokens")),
        max_total_tokens=_optional_int(raw.get("max_total_tokens")),
        max_llm_calls=_optional_int(raw.get("max_llm_calls")),
        max_tool_calls=_optional_int(raw.get("max_tool_calls")),
        max_rag_invocations=_optional_int(raw.get("max_rag_invocations")),
        max_websearch_invocations=_optional_int(raw.get("max_websearch_invocations")),
        max_wall_time_seconds=_optional_float(raw.get("max_wall_time_seconds")),
        max_planner_iterations=_optional_int(raw.get("max_planner_iterations")),
        max_replans=_optional_int(raw.get("max_replans")),
    )


def _scope_to_json(scope: BudgetReservationScope) -> dict[str, bool]:
    return {
        "input_tokens": scope.input_tokens,
        "output_tokens": scope.output_tokens,
        "total_tokens": scope.total_tokens,
        "llm_calls": scope.llm_calls,
        "tool_calls": scope.tool_calls,
        "rag_invocations": scope.rag_invocations,
        "websearch_invocations": scope.websearch_invocations,
        "wall_time_seconds": scope.wall_time_seconds,
        "planner_iterations": scope.planner_iterations,
        "replans": scope.replans,
    }


def _scope_from_json(raw: object) -> BudgetReservationScope:
    if not isinstance(raw, dict):
        raise RunBudgetLedgerSnapshotError("budget reservation scope must be an object")
    return BudgetReservationScope(
        input_tokens=_require_bool(raw.get("input_tokens"), field="input_tokens"),
        output_tokens=_require_bool(raw.get("output_tokens"), field="output_tokens"),
        total_tokens=_require_bool(raw.get("total_tokens"), field="total_tokens"),
        llm_calls=_require_bool(raw.get("llm_calls"), field="llm_calls"),
        tool_calls=_require_bool(raw.get("tool_calls"), field="tool_calls"),
        rag_invocations=_require_bool(raw.get("rag_invocations"), field="rag_invocations"),
        websearch_invocations=_require_bool(
            raw.get("websearch_invocations"),
            field="websearch_invocations",
        ),
        wall_time_seconds=_require_bool(raw.get("wall_time_seconds"), field="wall_time_seconds"),
        planner_iterations=_require_bool(
            raw.get("planner_iterations"),
            field="planner_iterations",
        ),
        replans=_require_bool(raw.get("replans"), field="replans"),
    )


def _record_to_json(record: PersistedBudgetRecord) -> dict[str, object]:
    return {
        "execution_id": str(record.execution_id),
        "parent_execution_id": str(record.parent_execution_id),
        "mode": record.mode.value,
        "allowance": _usage_totals_to_json(record.allowance),
        "reserved_scope": _scope_to_json(record.reserved_scope),
        "consumed": _usage_totals_to_json(record.consumed),
        "child_reserved": _usage_totals_to_json(record.child_reserved),
        "released": record.released,
    }


def _record_from_json(raw: object) -> PersistedBudgetRecord:
    if not isinstance(raw, dict):
        raise RunBudgetLedgerSnapshotError("budget reservation record must be an object")
    mode_raw = raw.get("mode")
    if mode_raw == ExecutionBudgetAllocationMode.SHARED.value:
        mode = ExecutionBudgetAllocationMode.SHARED
    elif mode_raw == ExecutionBudgetAllocationMode.RESERVED.value:
        mode = ExecutionBudgetAllocationMode.RESERVED
    else:
        raise RunBudgetLedgerSnapshotError("budget reservation record mode is invalid")
    execution_raw = raw.get("execution_id")
    parent_raw = raw.get("parent_execution_id")
    if not isinstance(execution_raw, str) or not isinstance(parent_raw, str):
        raise RunBudgetLedgerSnapshotError("budget reservation record execution ids are invalid")
    released_raw = raw.get("released")
    if not isinstance(released_raw, bool):
        raise RunBudgetLedgerSnapshotError("budget reservation record released must be bool")
    return PersistedBudgetRecord(
        execution_id=validate_execution_id(execution_raw),
        parent_execution_id=validate_execution_id(parent_raw),
        mode=mode,
        allowance=_usage_totals_from_json(raw.get("allowance")),
        reserved_scope=_scope_from_json(raw.get("reserved_scope")),
        consumed=_usage_totals_from_json(raw.get("consumed")),
        child_reserved=_usage_totals_from_json(raw.get("child_reserved")),
        released=released_raw,
    )


def encode_run_budget_snapshot(snapshot: RunBudgetLedgerSnapshot) -> bytes:
    payload = {
        "schema_version": snapshot.schema_version,
        "attempt_id": str(snapshot.attempt_id),
        "root_limits": _run_budget_to_json(snapshot.root_limits),
        "root_shared_consumed": _usage_totals_to_json(snapshot.root_shared_consumed),
        "root_permanent_consumed": _usage_totals_to_json(snapshot.root_permanent_consumed),
        "records": [_record_to_json(record) for record in snapshot.records],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def decode_run_budget_snapshot(raw: bytes) -> RunBudgetLedgerSnapshot:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunBudgetLedgerSnapshotError("run budget snapshot is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RunBudgetLedgerSnapshotError("run budget snapshot root must be an object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, int):
        raise RunBudgetLedgerSnapshotError("run budget snapshot schema_version must be int")
    validate_snapshot_schema_version(schema_version)
    attempt_id = validate_snapshot_attempt_id(payload.get("attempt_id"))
    records_raw = payload.get("records")
    if not isinstance(records_raw, list):
        raise RunBudgetLedgerSnapshotError("run budget snapshot records must be a list")
    records = tuple(_record_from_json(item) for item in records_raw)
    return RunBudgetLedgerSnapshot(
        schema_version=schema_version,
        attempt_id=attempt_id,
        root_limits=_run_budget_from_json(payload.get("root_limits")),
        root_shared_consumed=_usage_totals_from_json(payload.get("root_shared_consumed")),
        root_permanent_consumed=_usage_totals_from_json(payload.get("root_permanent_consumed")),
        records=records,
    )


def _require_int(raw: object, *, field: str) -> int:
    if not isinstance(raw, int):
        raise RunBudgetLedgerSnapshotError(f"{field} must be int")
    return raw


def _require_float(raw: object, *, field: str) -> float:
    if not isinstance(raw, (int, float)):
        raise RunBudgetLedgerSnapshotError(f"{field} must be number")
    return float(raw)


def _require_bool(raw: object, *, field: str) -> bool:
    if not isinstance(raw, bool):
        raise RunBudgetLedgerSnapshotError(f"{field} must be bool")
    return raw


class KvRunBudgetPersistence(RunBudgetPersistence):
    """DistributedKVStore-backed per-Run budget ledger snapshots."""

    def __init__(self, kv_store: DistributedKVStore) -> None:
        self._kv_store = kv_store

    def load_snapshot(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        return self._kv_store.get(tenant_id=tenant_id, key=_kv_storage_key(run_id))

    def compare_and_swap_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        snapshot: RunBudgetLedgerSnapshot,
    ) -> bool:
        return self._kv_store.compare_and_set(
            tenant_id=tenant_id,
            key=_kv_storage_key(run_id),
            expected=expected,
            new_value=encode_run_budget_snapshot(snapshot),
        )


class DocumentStoreRunBudgetPersistence(RunBudgetPersistence):
    """ConditionalDocumentStore-backed per-Run budget ledger snapshots."""

    def __init__(self, document_store: ConditionalDocumentStore) -> None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "run budget persistence requires ConditionalDocumentStore",
            )
        self._document_store = document_store

    def load_snapshot(self, *, tenant_id: str, run_id: RunId) -> bytes | None:
        record = self._document_store.get(_document_partition(tenant_id), _document_row_key(run_id))
        if record is None:
            return None
        encoded = record.data.get("snapshot")
        if encoded is None:
            return None
        if not isinstance(encoded, str):
            raise RunBudgetPersistenceError("run budget document snapshot must be a string")
        return encoded.encode("utf-8")

    def compare_and_swap_snapshot(
        self,
        *,
        tenant_id: str,
        run_id: RunId,
        expected: bytes | None,
        snapshot: RunBudgetLedgerSnapshot,
    ) -> bool:
        partition_key = _document_partition(tenant_id)
        row_key = _document_row_key(run_id)
        replacement = DocumentRecord(
            partition_key=partition_key,
            row_key=row_key,
            data={"snapshot": encode_run_budget_snapshot(snapshot).decode("utf-8")},
        )
        if expected is None:
            return self._document_store.put_if_absent(replacement)
        existing = self._document_store.get(partition_key, row_key)
        if existing is None:
            return False
        current = self._snapshot_bytes_from_record(existing)
        if current != expected:
            return False
        return self._document_store.replace_if_match(
            expected=existing,
            replacement=replacement,
        )

    @staticmethod
    def _snapshot_bytes_from_record(record: DocumentRecord) -> bytes:
        encoded = record.data.get("snapshot")
        if not isinstance(encoded, str):
            raise RunBudgetPersistenceError("run budget document snapshot must be a string")
        return encoded.encode("utf-8")


def wire_run_budget_persistence(
    *,
    kv_store: DistributedKVStore | None = None,
    document_store: DocumentStore | None = None,
) -> RunBudgetPersistence:
    """Platform composition boundary: storage capability → budget persistence."""
    if kv_store is not None and document_store is not None:
        raise ValueError(
            "wire_run_budget_persistence accepts kv_store or document_store, not both",
        )
    if kv_store is not None:
        return KvRunBudgetPersistence(kv_store)
    if document_store is not None:
        if not isinstance(document_store, ConditionalDocumentStore):
            raise TypeError(
                "run budget persistence requires ConditionalDocumentStore",
            )
        return DocumentStoreRunBudgetPersistence(document_store)
    raise ValueError(
        "wire_run_budget_persistence requires kv_store or document_store",
    )


class DurableExecutionBudgetLedger:
    """ExecutionBudgetLedger wrapper that persists canonical state per RunId."""

    __slots__ = (
        "_attempt_id",
        "_inner",
        "_last_known_raw",
        "_persistence",
        "_run_id",
        "_tenant_id",
    )

    def __init__(
        self,
        *,
        inner: InMemoryExecutionBudgetLedger,
        persistence: RunBudgetPersistence,
        tenant_id: str,
        run_id: RunId,
        attempt_id: AttemptId,
        last_known_raw: bytes | None,
    ) -> None:
        self._inner = inner
        self._persistence = persistence
        self._tenant_id = tenant_id
        self._run_id = run_id
        self._attempt_id = attempt_id
        self._last_known_raw = last_known_raw

    def grant_child_budget(
        self,
        *,
        execution_id: ExecutionId,
        parent_execution_id: ExecutionId,
        decision: ChildBudgetAllocationDecision,
    ) -> ExecutionBudgetReservationGrant:
        grant = self._inner.grant_child_budget(
            execution_id=execution_id,
            parent_execution_id=parent_execution_id,
            decision=decision,
        )
        self._persist_current_state()
        return grant

    def release_child_budget(self, execution_id: ExecutionId) -> None:
        self._inner.release_child_budget(execution_id)
        self._persist_current_state()

    def ensure_shared_participant(
        self,
        execution_id: ExecutionId,
        *,
        parent_execution_id: ExecutionId,
    ) -> None:
        self._inner.ensure_shared_participant(
            execution_id,
            parent_execution_id=parent_execution_id,
        )
        self._persist_current_state()

    def consume_budget(self, execution_id: ExecutionId, amounts: BudgetUsageTotals) -> None:
        self._inner.consume_budget(execution_id, amounts)
        self._persist_current_state()

    def snapshot_root_available(self) -> RunBudget:
        return self._inner.snapshot_root_available()

    def snapshot_reservation_remaining(self, execution_id: ExecutionId) -> RunBudget:
        return self._inner.snapshot_reservation_remaining(execution_id)

    def _persist_current_state(self) -> None:
        snapshot = self._inner.export_snapshot(self._attempt_id)
        while True:
            if self._persistence.compare_and_swap_snapshot(
                tenant_id=self._tenant_id,
                run_id=self._run_id,
                expected=self._last_known_raw,
                snapshot=snapshot,
            ):
                self._last_known_raw = encode_run_budget_snapshot(snapshot)
                return
            self._reload_from_storage()


    def _reload_from_storage(self) -> None:
        raw = self._persistence.load_snapshot(tenant_id=self._tenant_id, run_id=self._run_id)
        if raw is None:
            raise RunBudgetPersistenceError("run budget state disappeared during persistence")
        snapshot = _decode_or_fail(raw)
        if snapshot.attempt_id != self._attempt_id:
            inner = create_execution_budget_ledger(snapshot.root_limits)
            inner.restore_snapshot(snapshot)
            inner.prepare_for_attempt_redelivery()
            self._inner = inner
            self._last_known_raw = raw
            return
        self._inner.restore_snapshot(snapshot)
        self._last_known_raw = raw


@dataclass(frozen=True, slots=True)
class DurableRunBudgetLedgerFactory:
    """Create durable per-Run ledgers backed by platform storage."""

    persistence: RunBudgetPersistence
    default_run_budget: RunBudget | None = None

    def create_ledger(
        self,
        run_budget: RunBudget | None = None,
        *,
        tenant_id: str | None = None,
        run_id: RunId | None = None,
        attempt_id: AttemptId | None = None,
    ) -> ExecutionBudgetLedger:
        if tenant_id is None or run_id is None or attempt_id is None:
            raise RunBudgetPersistenceError(
                "durable run budget ledger requires tenant_id, run_id, and attempt_id",
            )
        resolved_limits = run_budget if run_budget is not None else self.default_run_budget
        limits = resolved_limits if resolved_limits is not None else RunBudget()
        scoped_run_id = validate_run_id(run_id)
        scoped_attempt_id = validate_attempt_id(attempt_id)
        raw = self.persistence.load_snapshot(tenant_id=tenant_id, run_id=scoped_run_id)
        if raw is None:
            inner = create_execution_budget_ledger(limits)
            snapshot = inner.export_snapshot(scoped_attempt_id)
            if not self.persistence.compare_and_swap_snapshot(
                tenant_id=tenant_id,
                run_id=scoped_run_id,
                expected=None,
                snapshot=snapshot,
            ):
                return self.create_ledger(
                    run_budget,
                    tenant_id=tenant_id,
                    run_id=scoped_run_id,
                    attempt_id=scoped_attempt_id,
                )
            return DurableExecutionBudgetLedger(
                inner=inner,
                persistence=self.persistence,
                tenant_id=tenant_id,
                run_id=scoped_run_id,
                attempt_id=scoped_attempt_id,
                last_known_raw=encode_run_budget_snapshot(snapshot),
            )

        snapshot = _decode_or_fail(raw)
        inner = create_execution_budget_ledger(snapshot.root_limits)
        inner.restore_snapshot(snapshot)
        if snapshot.attempt_id != scoped_attempt_id:
            inner.prepare_for_attempt_redelivery()
            settled = inner.export_snapshot(scoped_attempt_id)
            if not self.persistence.compare_and_swap_snapshot(
                tenant_id=tenant_id,
                run_id=scoped_run_id,
                expected=raw,
                snapshot=settled,
            ):
                return self.create_ledger(
                    run_budget,
                    tenant_id=tenant_id,
                    run_id=scoped_run_id,
                    attempt_id=scoped_attempt_id,
                )
            raw = encode_run_budget_snapshot(settled)
        return DurableExecutionBudgetLedger(
            inner=inner,
            persistence=self.persistence,
            tenant_id=tenant_id,
            run_id=scoped_run_id,
            attempt_id=scoped_attempt_id,
            last_known_raw=raw,
        )


def create_durable_run_budget_ledger_factory(
    persistence: RunBudgetPersistence,
    run_budget: RunBudget | None = None,
) -> DurableRunBudgetLedgerFactory:
    return DurableRunBudgetLedgerFactory(
        persistence=persistence,
        default_run_budget=run_budget,
    )


def _decode_or_fail(raw: bytes) -> RunBudgetLedgerSnapshot:
    try:
        return decode_run_budget_snapshot(raw)
    except RunBudgetLedgerSnapshotError as exc:
        raise RunBudgetPersistenceError(str(exc)) from exc
