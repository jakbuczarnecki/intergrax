# © Artur Czarnecki. All rights reserved.

"""Completed-work load accounting for DIST-SCALE-X P2 physical qualification."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue

P2_TASK_PAYLOAD_BYTES = 1024
P2_MAX_IN_FLIGHT = 2
P2_TASK_COMPLETION_DRAIN_SECONDS = 120.0
P2_PER_TASK_ACTIVE_TIMEOUT_SECONDS = 90.0
P2_READ_SHARE_TARGET = 0.30
P2_READ_SHARE_TOLERANCE = 0.08


@dataclass(slots=True)
class OpenTask:
    handle: TaskHandle
    submitted_at: float
    kind: str
    tenant_id: str


@dataclass(slots=True)
class TaskLoadAccounting:
    submitted: int = 0
    completed: int = 0
    failed: int = 0
    timed_out: int = 0
    read_submitted: int = 0
    write_submitted: int = 0
    completion_latencies_ms: list[float] = field(default_factory=list)
    completion_latencies_by_tenant: dict[str, list[float]] = field(default_factory=dict)
    completions_by_tenant: dict[str, int] = field(default_factory=dict)
    max_in_flight_observed: int = 0

    @property
    def outstanding(self) -> int:
        return max(0, self.submitted - self.completed - self.failed - self.timed_out)

    def useful_throughput_per_sec(self, duration_s: float) -> float:
        if duration_s <= 0:
            return 0.0
        return self.completed / duration_s

    def assert_acceptance_reconciled(self) -> None:
        if self.outstanding != 0:
            raise AssertionError(f"unresolved tasks after drain: outstanding={self.outstanding}")
        if self.timed_out > 0:
            raise AssertionError(f"timed_out tasks: {self.timed_out}")
        if self.failed > 0:
            raise AssertionError(f"failed tasks: {self.failed}")
        if self.completed <= 0:
            raise AssertionError("no completed work recorded")


def _payload_for_kind(kind: str) -> bytes:
    body = f'{{"kind":"{kind}"}}'.encode()
    if len(body) >= P2_TASK_PAYLOAD_BYTES:
        return body[:P2_TASK_PAYLOAD_BYTES]
    pad = P2_TASK_PAYLOAD_BYTES - len(body)
    return body + (b"x" * pad)


def _poll_open(
    queue: DocumentStoreTaskQueue,
    open_tasks: list[OpenTask],
    accounting: TaskLoadAccounting,
    *,
    now: float,
) -> None:
    still_open: list[OpenTask] = []
    for item in open_tasks:
        status = queue.get_status(item.handle)
        if status is TaskStatus.SUCCEEDED:
            accounting.completed += 1
            latency_ms = (now - item.submitted_at) * 1000.0
            accounting.completion_latencies_ms.append(latency_ms)
            accounting.completion_latencies_by_tenant.setdefault(item.tenant_id, []).append(
                latency_ms,
            )
            accounting.completions_by_tenant[item.tenant_id] = (
                accounting.completions_by_tenant.get(item.tenant_id, 0) + 1
            )
            continue
        if status is TaskStatus.FAILED:
            accounting.failed += 1
            continue
        if now - item.submitted_at > P2_PER_TASK_ACTIVE_TIMEOUT_SECONDS:
            accounting.timed_out += 1
            continue
        still_open.append(item)
    open_tasks[:] = still_open


def run_document_store_completed_work_load(
    queue: DocumentStoreTaskQueue,
    *,
    duration_s: float,
    tenants: tuple[str, ...],
    task_name: str,
    hot_tenant: str | None = None,
    hot_share: float = 0.0,
    max_in_flight: int = P2_MAX_IN_FLIGHT,
    worker_health: Callable[[], None] | None = None,
) -> TaskLoadAccounting:
    """Drive load with in-flight cap; throughput is based on terminal SUCCEEDED work only."""
    if not tenants:
        raise ValueError("tenants required")
    accounting = TaskLoadAccounting()
    open_tasks: list[OpenTask] = []
    deadline = time.monotonic() + duration_s
    tick = 0
    while time.monotonic() < deadline:
        if worker_health is not None:
            worker_health()
        now = time.monotonic()
        _poll_open(queue, open_tasks, accounting, now=now)
        accounting.max_in_flight_observed = max(
            accounting.max_in_flight_observed,
            len(open_tasks),
        )
        if len(open_tasks) >= max_in_flight:
            time.sleep(0.05)
            continue
        tick += 1
        use_hot = hot_tenant is not None and (tick % 10) < int(10 * hot_share)
        tenant = hot_tenant if use_hot and hot_tenant else tenants[tick % len(tenants)]
        kind = "read" if tick % 10 < 3 else "write"
        if kind == "read":
            accounting.read_submitted += 1
        else:
            accounting.write_submitted += 1
        handle = queue.enqueue(
            TaskRequest(
                tenant_id=tenant,
                run_id=f"run-{uuid.uuid4().hex[:12]}",
                task_name=task_name,
                payload=_payload_for_kind(kind),
                idempotency_key=None,
            )
        )
        accounting.submitted += 1
        open_tasks.append(
            OpenTask(handle=handle, submitted_at=time.monotonic(), kind=kind, tenant_id=tenant),
        )
        time.sleep(0.02)
    drain_deadline = time.monotonic() + P2_TASK_COMPLETION_DRAIN_SECONDS
    while open_tasks and time.monotonic() < drain_deadline:
        if worker_health is not None:
            worker_health()
        _poll_open(queue, open_tasks, accounting, now=time.monotonic())
        time.sleep(0.02)
    if open_tasks:
        accounting.timed_out += len(open_tasks)
        open_tasks.clear()
    if accounting.submitted >= 30:
        read_share = accounting.read_submitted / float(accounting.submitted)
        low = P2_READ_SHARE_TARGET - P2_READ_SHARE_TOLERANCE
        high = P2_READ_SHARE_TARGET + P2_READ_SHARE_TOLERANCE
        if not (low <= read_share <= high):
            raise AssertionError(
                f"task mix outside WL-1 envelope: read_share={read_share:.3f} submitted={accounting.submitted}",
            )
    return accounting


def pending_task_count(queue: DocumentStoreTaskQueue, tenant_id: str) -> int:
    rows = queue.list_tasks(tenant_id, limit=500)
    return sum(
        1
        for row in rows
        if row.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
    )


def assert_cross_tenant_queue_fail_closed(
    queue: DocumentStoreTaskQueue,
    *,
    owner_tenant: str,
    victim_tenant: str,
) -> None:
    """Cross-tenant handle must not surface another tenant's terminal state or payload."""
    rows = [r for r in queue.list_tasks(owner_tenant, limit=50) if r.status is TaskStatus.SUCCEEDED]
    if not rows:
        raise AssertionError("need at least one succeeded task for cross-tenant proof")
    task_id = rows[0].task_id
    foreign = TaskHandle(
        task_id=task_id,
        provider=rows[0].provider,
        tenant_id=victim_tenant,
    )
    status = queue.get_status(foreign)
    if status is TaskStatus.SUCCEEDED:
        raise AssertionError("cross-tenant get_status must not report SUCCEEDED")
    result = queue.get_result(foreign)
    if result is not None and result.status is TaskStatus.SUCCEEDED:
        raise AssertionError("cross-tenant get_result must not expose succeeded payload")
