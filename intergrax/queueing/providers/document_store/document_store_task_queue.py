# © Artur Czarnecki. All rights reserved.

"""Durable TaskQueue backed by the provider-neutral DocumentStore contract.

Solves single-host product failure where HTTP-owned ``asyncio.create_task`` work
is lost on process restart. Reuses DocumentStore (MongoDB / in-memory / …) rather
than inventing an application-specific queue. Belongs below Tier-3 because any
product host can enqueue through the shared MessageBus / TaskQueue contract.

Concurrency: over a ``ConditionalDocumentStore`` every state transition is a
compare-and-set (``replace_if_match``) and an idempotency key is reserved with
``put_if_absent``, so several queue instances (or hosts) sharing one store never claim a
task twice and never create two tasks for one key. Over a plain ``DocumentStore`` the
per-instance lock is the only exclusion: that mode is single-instance by contract.
Terminal states (``SUCCEEDED``, ``FAILED``, ``DEAD_LETTERED``) are sticky.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.contracts.idempotency_store import (
    ClaimOutcome,
    IdempotencyStore,
    InvocationClaim,
)
from intergrax.contracts.lease_claim import LeaseOwnership, StaleClaimError
from intergrax.integrations.contracts.document_store import (
    ConditionalDocumentStore,
    DocumentRecord,
    DocumentStore,
)
from intergrax.queueing.contracts.task_queue import (
    DeadLetterReason,
    DeadLetterRecord,
    TaskExecutionIdentity,
    TaskHandle,
    TaskQueue,
    TaskRequest,
    TaskResult,
    TaskStatus,
    TaskSummary,
    is_terminal_task_status,
)
from intergrax.queueing.providers.document_store.task_row_lease import (
    invocation_claim_for_new_attempt,
    invocation_claim_from_row,
    read_claim_from_row,
    row_claim_is_live,
    row_claim_matches,
    task_row_ledger_key,
    write_claim_to_row,
)
from intergrax.queueing.worker.retry_backoff import calculate_retry_countdown
from intergrax.queueing.worker.retry_policy import RetryPolicy
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.event_catalog import phase_for_event
from intergrax.runtime.events.payloads.canonical import TaskLifecyclePayloadV1
from intergrax.runtime.events.payloads.spine_families import TaskProgressPayloadV1
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType

DOCUMENT_STORE_TASK_QUEUE_PROVIDER = "document_store"
_PARTITION_PREFIX = "intergrax.task_queue.v1"
_PENDING_INDEX_PARTITION = f"{_PARTITION_PREFIX}:__pending_index__"
_RUNNING_INDEX_PARTITION = f"{_PARTITION_PREFIX}:__running_index__"
# Retry back-off rows wait here, keyed by availability time, so they never hide fresh work (F-D7-08).
_DELAYED_INDEX_PARTITION = f"{_PARTITION_PREFIX}:__delayed_index__"
_IDEMPOTENCY_PREFIX = "idempotency:"
_CAS_ATTEMPTS = 8
_PENDING_SCAN_PAGES = 32  # upper bound of one claim call: pages x max(4 * limit, 32) index rows

logger = logging.getLogger(__name__)


def _partition(tenant_id: str) -> str:
    return f"{_PARTITION_PREFIX}:{tenant_id}"


def _index_row(tenant_id: str, task_id: str) -> str:
    return f"{tenant_id}:{task_id}"


def _int_field(row: dict[str, object], key: str) -> int:
    value = row.get(key)
    return int(value) if isinstance(value, int | float | str) and value else 0


def _float_field(row: dict[str, object], key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, int | float | str) and value else 0.0


def _delayed_row(available_at: float, tenant_id: str, task_id: str) -> str:
    # Zero-padded so that the lexicographic row-key order is the availability order.
    return f"{available_at:017.6f}|{tenant_id}:{task_id}"


@dataclass(frozen=True)
class _CasResult:
    kind: Literal["missing", "rejected", "applied"]
    previous: dict[str, object] | None = None
    updated: dict[str, object] | None = None


@dataclass(frozen=True)
class TaskRowClaimContext:
    """Fencing context applied during ``claim_pending`` (existing idempotency ledger)."""

    owner_id: str
    lease_seconds: int = 300


@dataclass(frozen=True)
class DocumentStoreWakeClaim:
    """Successful durable-row claim for a targeted Celery wake delivery."""

    request: TaskRequest
    presented_claim: LeaseOwnership


class DocumentStoreWakeClaimNotGrantedError(RuntimeError):
    """The wake target row could not be claimed; the broker message must not be acknowledged."""


class TaskDeliveryMismatchError(ValueError):
    """Delivery tenant or identity does not match the durable task row."""


def _assert_delivery_matches_row(
    row: dict[str, object],
    *,
    tenant_id: str,
    identity: TaskExecutionIdentity | None,
) -> None:
    row_tenant = str(row.get("tenant_id") or "").strip()
    if row_tenant != tenant_id:
        raise TaskDeliveryMismatchError(
            f"delivery tenant {tenant_id!r} does not match row tenant {row_tenant!r}",
        )
    if identity is None:
        return
    raw_identity = row.get("identity")
    if not isinstance(raw_identity, dict):
        return
    row_identity = TaskExecutionIdentity.from_dict(raw_identity)
    for field in ("task_id", "run_id", "attempt_id", "execution_id"):
        delivered = getattr(identity, field, None)
        stored = getattr(row_identity, field, None)
        if delivered is not None and stored is not None and delivered != stored:
            raise TaskDeliveryMismatchError(
                f"delivery identity field {field} does not match stored row",
            )


def _require_live_row_claim(
    row: dict[str, object],
    *,
    presented: LeaseOwnership | None,
) -> InvocationClaim:
    parsed = read_claim_from_row(row)
    if parsed is None:
        raise StaleClaimError("task row has no active claim fence")
    if presented is None or not row_claim_matches(row, presented):
        raise StaleClaimError("task row claim fence does not match presented ownership")
    if not row_claim_is_live(row):
        raise StaleClaimError("task row claim lease expired")
    tenant_id = str(row.get("tenant_id") or "").strip()
    claim = invocation_claim_from_row(row, tenant_id=tenant_id)
    if claim is None:
        raise StaleClaimError("task row claim could not be decoded")
    return claim


def _queue_lifecycle_payload(
    *,
    event_type: RuntimeEventType,
    document_task_id: str,
    tenant_id: str,
) -> dict[str, object]:
    """Typed spine envelope plus the queue row identity the bus payload must carry."""
    if event_type is RuntimeEventType.TASK_PROGRESS:
        typed = TaskProgressPayloadV1(
            progress_kind="queue_claim",
            message=document_task_id,
            auxiliary={
                "document_task_id": document_task_id,
                "tenant_id": tenant_id,
            },
        )
    else:
        task_state = "FAILED" if event_type is RuntimeEventType.TASK_FAILED else "SUCCEEDED"
        typed = TaskLifecyclePayloadV1(
            task_state=task_state,
            message=document_task_id,
            source="document_store_task_queue",
        )
    envelope = typed.to_envelope()
    envelope["document_task_id"] = document_task_id
    envelope["tenant_id"] = tenant_id
    return envelope


class DocumentStoreTaskQueue(TaskQueue):
    """Crash-safe local MessageBus using DocumentStore as the durable log."""

    def __init__(
        self,
        document_store: DocumentStore,
        *,
        idempotency_store: IdempotencyStore | None = None,
        default_lease_seconds: int = 300,
        runtime_event_bus: RuntimeEventBus | None = None,
    ) -> None:
        self._store = document_store
        self._idempotency_store = idempotency_store
        self._default_lease_seconds = default_lease_seconds
        self._runtime_event_bus = runtime_event_bus
        self._lock = threading.RLock()
        self._conditional: ConditionalDocumentStore | None = (
            document_store if isinstance(document_store, ConditionalDocumentStore) else None
        )

    def assert_delivery_matches(
        self,
        handle: TaskHandle,
        *,
        tenant_id: str,
        identity: TaskExecutionIdentity | None,
    ) -> None:
        row = self._load(handle)
        if row is None:
            raise TaskDeliveryMismatchError(f"task row {handle.task_id!r} is missing")
        _assert_delivery_matches_row(row, tenant_id=tenant_id, identity=identity)

    @property
    def document_store(self) -> DocumentStore:
        return self._store

    @staticmethod
    def _canonical_identity_from_row(row: dict[str, object]) -> TaskExecutionIdentity | None:
        raw_identity = row.get("identity")
        if not isinstance(raw_identity, dict):
            return None
        try:
            identity = TaskExecutionIdentity.from_dict(raw_identity)
        except ValueError:
            return None
        if not identity.has_canonical_execution_ids():
            return None
        return identity

    def _emit_queue_lifecycle_event(
        self,
        row: dict[str, object],
        *,
        event_type: RuntimeEventType,
    ) -> None:
        bus = self._runtime_event_bus
        if bus is None:
            return
        identity = self._canonical_identity_from_row(row)
        if identity is None:
            return
        tenant_id = str(row.get("tenant_id") or "").strip()
        document_task_id = str(row.get("task_id") or "").strip()
        if not tenant_id or not document_task_id:
            return
        if identity.task_id is None or identity.run_id is None:
            return
        if identity.attempt_id is None or identity.execution_id is None:
            return
        phase = phase_for_event(event_type)
        if phase is None:
            return
        event = RuntimeEvent(
            tenant_id=tenant_id,
            task_id=TaskId(identity.task_id),
            run_id=RunId(identity.run_id),
            attempt_id=AttemptId(identity.attempt_id),
            execution_id=ExecutionId(identity.execution_id),
            event_type=event_type,
            phase=phase,
            payload=_queue_lifecycle_payload(
                event_type=event_type,
                document_task_id=document_task_id,
                tenant_id=tenant_id,
            ),
        )
        bus.record(event, tenant_id=tenant_id)

    def enqueue(self, request: TaskRequest) -> TaskHandle:
        with self._lock:
            task_id = f"dstq_{uuid.uuid4().hex}"
            if request.idempotency_key:
                existing_id = self._reserve_idempotency(request, task_id)
                if existing_id is not None:
                    return TaskHandle(
                        task_id=existing_id,
                        provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                        tenant_id=request.tenant_id,
                    )

            now = time.time()
            payload = {
                "task_id": task_id,
                "tenant_id": request.tenant_id,
                "run_id": request.run_id,
                "task_name": request.task_name,
                "payload_base64": base64.b64encode(request.payload).decode("ascii"),
                "idempotency_key": request.idempotency_key,
                "status": TaskStatus.PENDING.value,
                "attempts": 0,
                "created_at_epoch": now,
                "updated_at_epoch": now,
                "available_at_epoch": now,
                "error_message": None,
                "output_base64": None,
                "identity": (request.identity.to_dict() if request.identity is not None else None),
                "dead_letter": None,
            }
            self._store.put(
                DocumentRecord(
                    partition_key=_partition(request.tenant_id),
                    row_key=task_id,
                    data=payload,
                )
            )
            self._store.put(
                DocumentRecord(
                    partition_key=_PENDING_INDEX_PARTITION,
                    row_key=_index_row(request.tenant_id, task_id),
                    data={"tenant_id": request.tenant_id, "task_id": task_id},
                )
            )
            if request.idempotency_key and self._conditional is None:
                self._store.put(
                    DocumentRecord(
                        partition_key=_partition(request.tenant_id),
                        row_key=f"{_IDEMPOTENCY_PREFIX}{request.idempotency_key}",
                        data={"task_id": task_id},
                    )
                )
            return TaskHandle(
                task_id=task_id,
                provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                tenant_id=request.tenant_id,
            )

    def _reserve_idempotency(self, request: TaskRequest, task_id: str) -> str | None:
        """Return the task id that already owns the key, or None when ``task_id`` now owns it.

        Plain store: lookup under the per-instance lock (the row is written by ``enqueue``).
        Conditional store: the key is reserved with ``put_if_absent`` before the task row is
        written, and a reservation whose task row never appeared (crash in between) is repaired
        with ``replace_if_match`` instead of being returned as a live handle.
        """
        key = request.idempotency_key or ""
        existing_id = self._lookup_idempotency(tenant_id=request.tenant_id, idempotency_key=key)
        conditional = self._conditional
        if conditional is None:
            return existing_id
        reservation = DocumentRecord(
            partition_key=_partition(request.tenant_id),
            row_key=f"{_IDEMPOTENCY_PREFIX}{key}",
            data={"task_id": task_id},
        )
        for _ in range(_CAS_ATTEMPTS):
            if conditional.put_if_absent(reservation):
                return None
            current = self._store.get(reservation.partition_key, reservation.row_key)
            if current is None:
                continue  # released between the two calls: try to reserve again
            owner = str(current.data.get("task_id") or "").strip()
            if owner and self._store.get(_partition(request.tenant_id), owner) is not None:
                return owner
            if conditional.replace_if_match(expected=current, replacement=reservation):
                logger.warning("repaired orphan idempotency reservation for tenant=%s", request.tenant_id)
                return None
        raise RuntimeError("document_store_task_queue_idempotency_contention")

    def get_status(self, handle: TaskHandle) -> TaskStatus:
        record = self._load(handle)
        if record is None:
            return TaskStatus.PENDING
        return TaskStatus(str(record["status"]))

    def get_status_if_present(self, handle: TaskHandle) -> TaskStatus | None:
        record = self._load(handle)
        if record is None:
            return None
        return TaskStatus(str(record["status"]))

    def get_result(self, handle: TaskHandle) -> TaskResult | None:
        record = self._load(handle)
        if record is None:
            return None
        status = TaskStatus(str(record["status"]))
        if not is_terminal_task_status(status):
            return None
        output: bytes | None = None
        encoded = record.get("output_base64")
        if isinstance(encoded, str) and encoded:
            output = base64.b64decode(encoded.encode("ascii"))
        error = record.get("error_message")
        dead_letter: DeadLetterRecord | None = None
        raw_dead_letter = record.get("dead_letter")
        if status is TaskStatus.DEAD_LETTERED and isinstance(raw_dead_letter, dict):
            dead_letter = DeadLetterRecord.from_dict(raw_dead_letter)
        return TaskResult(
            status=status,
            output=output,
            error_message=str(error) if error else None,
            attempts=int(record.get("attempts") or 0),
            dead_letter=dead_letter,
        )

    def list_tasks(
        self,
        tenant_id: str,
        *,
        limit: int = 100,
        status_filter: TaskStatus | None = None,
    ) -> list[TaskSummary]:
        rows = self._list_task_rows(tenant_id=tenant_id, limit=max(limit * 2, 100))
        summaries: list[TaskSummary] = []
        for row in rows:
            status = TaskStatus(str(row["status"]))
            if status_filter is not None and status is not status_filter:
                continue
            summaries.append(
                TaskSummary(
                    task_id=str(row["task_id"]),
                    tenant_id=tenant_id,
                    task_name=str(row["task_name"]),
                    status=status,
                    provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                )
            )
            if len(summaries) >= limit:
                break
        return summaries

    def cancel(self, handle: TaskHandle) -> bool:
        with self._lock:
            tenant_id = handle.tenant_id or ""

            def _cancel(row: dict[str, object]) -> dict[str, object] | None:
                if TaskStatus(str(row["status"])) is not TaskStatus.PENDING:
                    return None
                row["status"] = TaskStatus.FAILED.value
                row["error_message"] = "cancelled"
                row["updated_at_epoch"] = time.time()
                return row

            if not tenant_id or self._cas_update(tenant_id, handle.task_id, _cancel).kind != "applied":
                return False
            self._drop_indexes(tenant_id, handle.task_id)
            return True

    def claim_pending(
        self,
        *,
        tenant_id: str | None = None,
        limit: int = 8,
        claim_context: TaskRowClaimContext | None = None,
    ) -> list[tuple[TaskHandle, TaskRequest]]:
        """Atomically transition PENDING → RUNNING and return claimed work.

        Rows whose retry back-off has not elapsed live in the delayed index and are promoted
        when due, so they never hide fresh work. Pending-index rows written before that split
        (back-off rows that are not due) are relocated to the delayed index on the way.
        """
        with self._lock:
            claimed: list[tuple[TaskHandle, TaskRequest]] = []
            self._promote_due_delayed(limit)
            page_size = max(limit * 4, 32)
            cursor: str | None = None
            for _ in range(_PENDING_SCAN_PAGES):
                pending = self._store.query(_PENDING_INDEX_PARTITION, limit=page_size, cursor=cursor)
                for doc in pending.documents:
                    tid = str(doc.data.get("tenant_id") or "").strip()
                    task_id = str(doc.data.get("task_id") or "").strip()
                    if not tid or not task_id:
                        continue
                    if tenant_id is not None and tid != tenant_id:
                        continue
                    row = self._load(
                        TaskHandle(task_id=task_id, provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER, tenant_id=tid)
                    )
                    if row is None or TaskStatus(str(row["status"])) is not TaskStatus.PENDING:
                        self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tid, task_id))
                        continue
                    available_at = _float_field(row, "available_at_epoch")
                    if available_at > time.time():
                        self._relocate_to_delayed(tid, task_id, available_at)
                        continue
                    ledger_claim: InvocationClaim | None = None
                    if claim_context is not None and self._idempotency_store is not None:
                        request_probe = self._request_from_row(tid, row)
                        ledger_key = task_row_ledger_key(
                            logical_task_name=request_probe.task_name,
                            idempotency_key=request_probe.idempotency_key,
                            task_id=task_id,
                        )
                        claim_result = self._idempotency_store.claim(
                            tenant_id=tid,
                            key=ledger_key,
                            owner_id=claim_context.owner_id,
                            lease_seconds=claim_context.lease_seconds,
                        )
                        if claim_result.outcome is ClaimOutcome.BLOCKED_ACTIVE:
                            continue
                        if claim_result.outcome is ClaimOutcome.UNCERTAIN:
                            ledger_claim = None
                        elif claim_result.outcome is ClaimOutcome.REPLAY_COMPLETED:
                            self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tid, task_id))
                            continue
                        else:
                            ledger_claim = claim_result.claim
                            if ledger_claim is None:
                                continue

                    def _claim(row: dict[str, object]) -> dict[str, object] | None:
                        updated = DocumentStoreTaskQueue._claim_mutation(row)
                        if updated is None:
                            return None
                        if claim_context is not None:
                            ledger_key = task_row_ledger_key(
                                logical_task_name=str(row["task_name"]),
                                idempotency_key=(str(row["idempotency_key"]) if row.get("idempotency_key") else None),
                                task_id=task_id,
                            )
                            claim_to_write = ledger_claim or invocation_claim_for_new_attempt(
                                updated,
                                tenant_id=tid,
                                owner_id=claim_context.owner_id,
                                lease_seconds=claim_context.lease_seconds,
                                ledger_key=ledger_key,
                            )
                            write_claim_to_row(updated, claim=claim_to_write, ledger_key=ledger_key)
                        return updated

                    result = self._cas_update(tid, task_id, _claim)
                    if result.kind != "applied" or result.updated is None:
                        if ledger_claim is not None and self._idempotency_store is not None:
                            ledger_key = task_row_ledger_key(
                                logical_task_name=str(row.get("task_name") or ""),
                                idempotency_key=(
                                    str(row.get("idempotency_key")) if row.get("idempotency_key") else None
                                ),
                                task_id=task_id,
                            )
                            try:
                                self._idempotency_store.abandon_pre_effect_with_claim(
                                    tenant_id=tid,
                                    key=ledger_key,
                                    claim=ledger_claim,
                                )
                            except StaleClaimError:
                                pass
                        continue  # another instance won the compare-and-set
                    self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tid, task_id))
                    self._store.put(
                        DocumentRecord(
                            partition_key=_RUNNING_INDEX_PARTITION,
                            row_key=_index_row(tid, task_id),
                            data={"tenant_id": tid, "task_id": task_id},
                        )
                    )
                    handle = TaskHandle(
                        task_id=task_id,
                        provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                        tenant_id=tid,
                    )
                    self._emit_queue_lifecycle_event(result.updated, event_type=RuntimeEventType.TASK_PROGRESS)
                    claimed.append((handle, self._request_from_row(tid, result.updated)))
                    if len(claimed) >= limit:
                        return claimed
                cursor = pending.next_cursor
                if not cursor:
                    break
            return claimed

    def claim_task_for_wake(
        self,
        handle: TaskHandle,
        *,
        claim_context: TaskRowClaimContext,
    ) -> DocumentStoreWakeClaim | None:
        """Transition one pending row to running for a targeted Celery wake."""
        with self._lock:
            tenant_id = handle.tenant_id or ""
            task_id = handle.task_id
            row = self._load(handle)
            if row is None:
                return None
            if TaskStatus(str(row["status"])) is not TaskStatus.PENDING:
                return None
            if _float_field(row, "available_at_epoch") > time.time():
                return None
            ledger_claim: InvocationClaim | None = None
            if self._idempotency_store is not None:
                request_probe = self._request_from_row(tenant_id, row)
                ledger_key = task_row_ledger_key(
                    logical_task_name=request_probe.task_name,
                    idempotency_key=request_probe.idempotency_key,
                    task_id=task_id,
                )
                claim_result = self._idempotency_store.claim(
                    tenant_id=tenant_id,
                    key=ledger_key,
                    owner_id=claim_context.owner_id,
                    lease_seconds=claim_context.lease_seconds,
                )
                if claim_result.outcome is ClaimOutcome.BLOCKED_ACTIVE:
                    return None
                if claim_result.outcome is ClaimOutcome.UNCERTAIN:
                    ledger_claim = None
                elif claim_result.outcome is ClaimOutcome.REPLAY_COMPLETED:
                    self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tenant_id, task_id))
                    return None
                else:
                    ledger_claim = claim_result.claim
                    if ledger_claim is None:
                        return None

            def _claim(row: dict[str, object]) -> dict[str, object] | None:
                updated = DocumentStoreTaskQueue._claim_mutation(row)
                if updated is None:
                    return None
                ledger_key = task_row_ledger_key(
                    logical_task_name=str(row["task_name"]),
                    idempotency_key=(str(row["idempotency_key"]) if row.get("idempotency_key") else None),
                    task_id=task_id,
                )
                claim_to_write = ledger_claim or invocation_claim_for_new_attempt(
                    updated,
                    tenant_id=tenant_id,
                    owner_id=claim_context.owner_id,
                    lease_seconds=claim_context.lease_seconds,
                    ledger_key=ledger_key,
                )
                write_claim_to_row(updated, claim=claim_to_write, ledger_key=ledger_key)
                return updated

            result = self._cas_update(tenant_id, task_id, _claim)
            if result.kind != "applied" or result.updated is None:
                if ledger_claim is not None and self._idempotency_store is not None:
                    ledger_key = task_row_ledger_key(
                        logical_task_name=str(row.get("task_name") or ""),
                        idempotency_key=(str(row.get("idempotency_key")) if row.get("idempotency_key") else None),
                        task_id=task_id,
                    )
                    try:
                        self._idempotency_store.abandon_pre_effect_with_claim(
                            tenant_id=tenant_id,
                            key=ledger_key,
                            claim=ledger_claim,
                        )
                    except StaleClaimError:
                        pass
                return None
            self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tenant_id, task_id))
            self._store.put(
                DocumentRecord(
                    partition_key=_RUNNING_INDEX_PARTITION,
                    row_key=_index_row(tenant_id, task_id),
                    data={"tenant_id": tenant_id, "task_id": task_id},
                )
            )
            request = self._request_from_row(tenant_id, result.updated)
            self._emit_queue_lifecycle_event(result.updated, event_type=RuntimeEventType.TASK_PROGRESS)
            row_claim = invocation_claim_from_row(result.updated, tenant_id=tenant_id)
            if row_claim is None:
                return None
            presented = LeaseOwnership(
                owner_id=row_claim.owner_id,
                lease_expires_at=row_claim.lease_expires_at,
                fence=row_claim.fence,
            )
            return DocumentStoreWakeClaim(request=request, presented_claim=presented)

    @staticmethod
    def _claim_mutation(row: dict[str, object]) -> dict[str, object] | None:
        if TaskStatus(str(row["status"])) is not TaskStatus.PENDING:
            return None
        if _float_field(row, "available_at_epoch") > time.time():
            return None
        row["status"] = TaskStatus.RUNNING.value
        row["attempts"] = _int_field(row, "attempts") + 1
        row["updated_at_epoch"] = time.time()
        return row

    def _promote_due_delayed(self, limit: int) -> None:
        upper_bound = f"{time.time():017.6f}|\U0010ffff"
        due = self._store.query(
            _DELAYED_INDEX_PARTITION,
            limit=max(limit * 4, 32),
            row_key_upper_bound=upper_bound,
        )
        for doc in due.documents:
            self._store.put(
                DocumentRecord(
                    partition_key=_PENDING_INDEX_PARTITION,
                    row_key=_index_row(str(doc.data.get("tenant_id")), str(doc.data.get("task_id"))),
                    data={"tenant_id": doc.data.get("tenant_id"), "task_id": doc.data.get("task_id")},
                )
            )
            self._store.delete(_DELAYED_INDEX_PARTITION, doc.row_key)

    def _put_delayed(self, tenant_id: str, task_id: str, available_at: float) -> None:
        self._store.put(
            DocumentRecord(
                partition_key=_DELAYED_INDEX_PARTITION,
                row_key=_delayed_row(available_at, tenant_id, task_id),
                data={"tenant_id": tenant_id, "task_id": task_id},
            )
        )

    def _relocate_to_delayed(self, tenant_id: str, task_id: str, available_at: float) -> None:
        self._put_delayed(tenant_id, task_id, available_at)
        self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tenant_id, task_id))

    def mark_succeeded(
        self,
        handle: TaskHandle,
        *,
        output: bytes | None = None,
        presented_claim: LeaseOwnership | None = None,
    ) -> None:
        def _succeed(row: dict[str, object]) -> dict[str, object] | None:
            if self._standing_terminal(handle, row, "mark_succeeded"):
                return None
            if TaskStatus(str(row["status"])) is not TaskStatus.RUNNING:
                return None
            if read_claim_from_row(row) is not None:
                _require_live_row_claim(row, presented=presented_claim)
            row["status"] = TaskStatus.SUCCEEDED.value
            row["output_base64"] = base64.b64encode(output).decode("ascii") if output else None
            row["error_message"] = None
            row["updated_at_epoch"] = time.time()
            return row

        with self._lock:
            tenant_id = handle.tenant_id or ""
            result = self._cas_update(tenant_id, handle.task_id, _succeed)
            if result.kind == "applied":
                self._drop_indexes(tenant_id, handle.task_id)
                if result.updated is not None:
                    self._emit_queue_lifecycle_event(
                        result.updated,
                        event_type=RuntimeEventType.TASK_COMPLETED,
                    )

    def mark_failed(
        self,
        handle: TaskHandle,
        *,
        error_message: str,
        presented_claim: LeaseOwnership | None = None,
    ) -> None:
        def _fail(row: dict[str, object]) -> dict[str, object] | None:
            if self._standing_terminal(handle, row, "mark_failed"):
                return None
            if TaskStatus(str(row["status"])) is not TaskStatus.RUNNING:
                return None
            if read_claim_from_row(row) is not None:
                _require_live_row_claim(row, presented=presented_claim)
            row["status"] = TaskStatus.FAILED.value
            row["error_message"] = error_message[:500]
            row["updated_at_epoch"] = time.time()
            return row

        with self._lock:
            tenant_id = handle.tenant_id or ""
            result = self._cas_update(tenant_id, handle.task_id, _fail)
            if result.kind == "applied":
                self._drop_indexes(tenant_id, handle.task_id)
                if result.updated is not None:
                    self._emit_queue_lifecycle_event(
                        result.updated,
                        event_type=RuntimeEventType.TASK_FAILED,
                    )

    def mark_dead_lettered(
        self,
        handle: TaskHandle,
        *,
        reason: DeadLetterReason,
        error_message: str,
    ) -> None:
        """Move a task to the terminal ``DEAD_LETTERED`` state with a typed reason."""

        def _dead_letter(row: dict[str, object]) -> dict[str, object] | None:
            if self._standing_terminal(handle, row, "mark_dead_lettered"):
                return None
            return self._dead_letter_fields(row, reason=reason, error_message=error_message)

        with self._lock:
            tenant_id = handle.tenant_id or ""
            result = self._cas_update(tenant_id, handle.task_id, _dead_letter)
            if result.kind == "applied":
                self._drop_indexes(tenant_id, handle.task_id)
                if result.updated is not None:
                    self._emit_queue_lifecycle_event(
                        result.updated,
                        event_type=RuntimeEventType.TASK_FAILED,
                    )

    def retry_or_dead_letter(
        self,
        handle: TaskHandle,
        *,
        error_message: str,
        retry_policy: RetryPolicy,
        presented_claim: LeaseOwnership | None = None,
    ) -> TaskStatus:
        """Settle a failed delivery against the retry budget.

        ``attempts`` counts deliveries (incremented at claim). While
        ``attempts <= max_retries`` the task returns to ``PENDING`` after the policy
        back-off; once the budget is spent it becomes terminal ``DEAD_LETTERED``. A task
        that already reached a terminal status keeps it (the standing status is returned).
        """

        def _settle(row: dict[str, object]) -> dict[str, object] | None:
            if self._standing_terminal(handle, row, "retry_or_dead_letter"):
                return None
            if TaskStatus(str(row["status"])) is not TaskStatus.RUNNING:
                return None
            if read_claim_from_row(row) is not None:
                _require_live_row_claim(row, presented=presented_claim)
            return self._settled_fields(
                row,
                error_message=error_message,
                max_retries=retry_policy.max_retries,
                retry_policy=retry_policy,
            )

        with self._lock:
            tenant_id = handle.tenant_id or ""
            result = self._cas_update(tenant_id, handle.task_id, _settle)
            if result.kind == "missing":
                return TaskStatus.PENDING
            if result.kind == "rejected" and result.previous is not None:
                return TaskStatus(str(result.previous["status"]))
            assert result.updated is not None
            status = self._after_settle(tenant_id, result.updated)
            if status is TaskStatus.DEAD_LETTERED:
                self._emit_queue_lifecycle_event(
                    result.updated,
                    event_type=RuntimeEventType.TASK_FAILED,
                )
            return status

    @staticmethod
    def _standing_terminal(handle: TaskHandle, row: dict[str, object], operation: str) -> bool:
        """True (and logged) when the row already holds a terminal status: terminal is sticky."""
        status = TaskStatus(str(row["status"]))
        if is_terminal_task_status(status):
            logger.warning(
                "%s ignored for task %s: already terminal (%s)",
                operation,
                handle.task_id,
                status.value,
            )
            return True
        return False

    def _settled_fields(
        self,
        record: dict[str, object],
        *,
        error_message: str,
        max_retries: int,
        retry_policy: RetryPolicy | None,
    ) -> dict[str, object]:
        attempts = _int_field(record, "attempts")
        if attempts > max_retries:
            return self._dead_letter_fields(
                record,
                reason=DeadLetterReason.RETRY_BUDGET_EXHAUSTED,
                error_message=error_message,
            )
        countdown = (
            calculate_retry_countdown(
                policy=retry_policy,
                current_retries=max(attempts - 1, 0),
            )
            if retry_policy is not None
            else 0.0
        )
        now = time.time()
        record["status"] = TaskStatus.PENDING.value
        record["error_message"] = error_message[:500]
        record["updated_at_epoch"] = now
        record["available_at_epoch"] = now + countdown
        return record

    def _after_settle(self, tenant_id: str, record: dict[str, object]) -> TaskStatus:
        """Index maintenance after a settle was applied; returns the new status."""
        task_id = str(record["task_id"])
        status = TaskStatus(str(record["status"]))
        if status is TaskStatus.DEAD_LETTERED:
            self._drop_indexes(tenant_id, task_id)
            return status
        self._store.delete(_RUNNING_INDEX_PARTITION, _index_row(tenant_id, task_id))
        available_at = _float_field(record, "available_at_epoch")
        if available_at > time.time():
            self._put_delayed(tenant_id, task_id, available_at)
        else:
            self._store.put(
                DocumentRecord(
                    partition_key=_PENDING_INDEX_PARTITION,
                    row_key=_index_row(tenant_id, task_id),
                    data={"tenant_id": tenant_id, "task_id": task_id},
                )
            )
        return TaskStatus.PENDING

    @staticmethod
    def _dead_letter_fields(
        record: dict[str, object],
        *,
        reason: DeadLetterReason,
        error_message: str,
    ) -> dict[str, object]:
        message = error_message[:500]
        record["status"] = TaskStatus.DEAD_LETTERED.value
        record["error_message"] = message
        record["dead_letter"] = DeadLetterRecord(
            reason=reason,
            attempts=_int_field(record, "attempts"),
            last_error=message,
        ).to_dict()
        record["updated_at_epoch"] = time.time()
        return record

    def _cas_update(
        self,
        tenant_id: str,
        task_id: str,
        mutate: Callable[[dict[str, object]], dict[str, object] | None],
    ) -> _CasResult:
        """Read-modify-write one task row; a compare-and-set on a conditional store.

        ``mutate`` returns the new row, or None to reject the transition (the row is left
        untouched). A lost compare-and-set re-reads and re-evaluates, so a rejected
        transition is decided on the freshest row.
        """
        if not tenant_id:
            return _CasResult("missing")
        partition = _partition(tenant_id)
        for _ in range(_CAS_ATTEMPTS):
            record = self._store.get(partition, task_id)
            if record is None:
                return _CasResult("missing")
            previous = dict(record.data)
            updated = mutate(dict(previous))
            if updated is None:
                return _CasResult("rejected", previous=previous)
            replacement = DocumentRecord(partition_key=partition, row_key=task_id, data=updated)
            conditional = self._conditional
            if conditional is None:
                self._store.put(replacement)
                return _CasResult("applied", previous=previous, updated=updated)
            if conditional.replace_if_match(expected=record, replacement=replacement):
                return _CasResult("applied", previous=previous, updated=updated)
        raise RuntimeError("document_store_task_queue_cas_contention")

    def recover_interrupted_running(
        self,
        *,
        error_message: str = "interrupted_by_host_restart",
        max_retries: int | None = None,
    ) -> list[tuple[TaskHandle, TaskRequest]]:
        """Fail-closed recovery: RUNNING messages cannot remain permanently running.

        Returns the interrupted requests so product workers can update domain state.

        With ``max_retries=None`` (default) interrupted tasks become ``FAILED``. With a
        budget, a crash-looping (poison) task is redelivered until the budget is spent
        and then dead-lettered instead of being retried forever.

        Each row is settled by one compare-and-set against the row read in the same attempt
        (EX-L-03): a row that is no longer ``RUNNING`` (finished by a concurrent writer, or
        already recovered by another instance) is left untouched and not reported. Over a
        plain ``DocumentStore`` the write is an unconditional ``put`` and this stays the
        single-instance mode of the module docstring. Without a worker lease a crash and a
        slow writer that is still running are not told apart: this prevents the overwrite of
        a settled row, it does not fence a live one.
        """

        def _settle(row: dict[str, object]) -> dict[str, object] | None:
            if TaskStatus(str(row["status"])) is not TaskStatus.RUNNING:
                return None
            if row_claim_is_live(row):
                return None
            if max_retries is not None:
                return self._settled_fields(
                    row,
                    error_message=error_message,
                    max_retries=max_retries,
                    retry_policy=None,
                )
            row["status"] = TaskStatus.FAILED.value
            row["error_message"] = error_message
            row["updated_at_epoch"] = time.time()
            return row

        with self._lock:
            interrupted: list[tuple[TaskHandle, TaskRequest]] = []
            running = self._store.query(_RUNNING_INDEX_PARTITION, limit=500)
            for doc in running.documents:
                tid = str(doc.data.get("tenant_id") or "").strip()
                task_id = str(doc.data.get("task_id") or "").strip()
                if not tid or not task_id:
                    continue
                handle = TaskHandle(
                    task_id=task_id,
                    provider=DOCUMENT_STORE_TASK_QUEUE_PROVIDER,
                    tenant_id=tid,
                )
                result = self._cas_update(tid, task_id, _settle)
                if result.kind == "applied" and result.previous is not None and result.updated is not None:
                    interrupted.append((handle, self._request_from_row(tid, result.previous)))
                    if max_retries is not None:
                        self._after_settle(tid, result.updated)
                # Applied, missing, or observed in a non-RUNNING state: the index row is stale either way.
                self._store.delete(_RUNNING_INDEX_PARTITION, _index_row(tid, task_id))
            return interrupted

    @staticmethod
    def _request_from_row(tenant_id: str, row: dict[str, object]) -> TaskRequest:
        payload = base64.b64decode(str(row["payload_base64"]).encode("ascii"))
        raw_identity = row.get("identity")
        return TaskRequest(
            tenant_id=tenant_id,
            run_id=str(row["run_id"]),
            task_name=str(row["task_name"]),
            payload=payload,
            idempotency_key=(str(row["idempotency_key"]) if row.get("idempotency_key") else None),
            identity=(TaskExecutionIdentity.from_dict(raw_identity) if isinstance(raw_identity, dict) else None),
        )

    def _drop_indexes(self, tenant_id: str, task_id: str) -> None:
        self._store.delete(_PENDING_INDEX_PARTITION, _index_row(tenant_id, task_id))
        self._store.delete(_RUNNING_INDEX_PARTITION, _index_row(tenant_id, task_id))

    def _lookup_idempotency(self, *, tenant_id: str, idempotency_key: str) -> str | None:
        record = self._store.get(
            _partition(tenant_id),
            f"{_IDEMPOTENCY_PREFIX}{idempotency_key}",
        )
        if record is None:
            return None
        task_id = str(record.data.get("task_id") or "").strip()
        return task_id or None

    def _load(self, handle: TaskHandle) -> dict[str, object] | None:
        tenant_id = handle.tenant_id
        if not tenant_id:
            return None
        record = self._store.get(_partition(tenant_id), handle.task_id)
        if record is None:
            return None
        return dict(record.data)

    def _list_task_rows(self, *, tenant_id: str, limit: int) -> list[dict[str, object]]:
        result = self._store.query(_partition(tenant_id), limit=limit)
        rows: list[dict[str, object]] = []
        for doc in result.documents:
            if str(doc.row_key).startswith(_IDEMPOTENCY_PREFIX):
                continue
            if "task_id" not in doc.data or "task_name" not in doc.data:
                continue
            rows.append(dict(doc.data))
        rows.sort(key=lambda item: float(item.get("created_at_epoch") or 0.0))
        return rows

    def health(self):  # type: ignore[override]
        from intergrax.integrations.contracts.base import HealthStatus

        return HealthStatus(healthy=True, detail="document_store_task_queue")
