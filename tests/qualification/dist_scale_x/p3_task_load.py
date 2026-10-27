# © Artur Czarnecki. All rights reserved.

"""P3 completed-work load — 85% hot-tenant distribution (17/20 exact rational)."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue

from tests.qualification.dist_scale_x.p2_task_load import (
    OpenTask,
    P2_MAX_IN_FLIGHT,
    P2_TASK_COMPLETION_DRAIN_SECONDS,
    TaskLoadAccounting,
    _payload_for_kind,
    _poll_open,
    assert_cross_tenant_queue_fail_closed,
)

P3_HOT_NUMERATOR = 17
P3_HOT_DENOMINATOR = 20
P3_HOT_SHARE_TARGET = P3_HOT_NUMERATOR / P3_HOT_DENOMINATOR
P3_HOT_SHARE_TOLERANCE = 0.04


def p3_use_hot_bucket(tick: int) -> bool:
    return (tick % P3_HOT_DENOMINATOR) < P3_HOT_NUMERATOR


@dataclass(slots=True)
class HotShareLedger:
    submitted_hot: int = 0
    submitted_cold: int = 0

    @property
    def submitted_total(self) -> int:
        return self.submitted_hot + self.submitted_cold

    def measured_hot_share(self) -> float:
        if self.submitted_total == 0:
            return 0.0
        return self.submitted_hot / float(self.submitted_total)


def run_p3_hot_tenant_completed_work_load(
    queue: DocumentStoreTaskQueue,
    *,
    duration_s: float,
    cold_tenants: tuple[str, ...],
    hot_tenant: str,
    task_name: str,
    max_in_flight: int = P2_MAX_IN_FLIGHT,
    worker_health: Callable[[], None] | None = None,
) -> tuple[TaskLoadAccounting, HotShareLedger]:
    if not cold_tenants:
        raise ValueError("cold_tenants required")
    from intergrax.queueing.contracts.task_queue import TaskRequest

    accounting = TaskLoadAccounting()
    hot_ledger = HotShareLedger()
    open_tasks: list[OpenTask] = []
    deadline = time.monotonic() + duration_s
    tick = 0
    cold_list = list(cold_tenants)
    while time.monotonic() < deadline:
        if worker_health is not None:
            worker_health()
        now = time.monotonic()
        _poll_open(queue, open_tasks, accounting, now=now)
        if len(open_tasks) >= max_in_flight:
            time.sleep(0.05)
            continue
        tick += 1
        if p3_use_hot_bucket(tick):
            tenant = hot_tenant
            hot_ledger.submitted_hot += 1
        else:
            tenant = cold_list[tick % len(cold_list)]
            hot_ledger.submitted_cold += 1
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
        time.sleep(0.06)
    drain_deadline = time.monotonic() + P2_TASK_COMPLETION_DRAIN_SECONDS
    while open_tasks and time.monotonic() < drain_deadline:
        if worker_health is not None:
            worker_health()
        _poll_open(queue, open_tasks, accounting, now=time.monotonic())
        time.sleep(0.06)
    if open_tasks:
        accounting.timed_out += len(open_tasks)
        open_tasks.clear()
    share = hot_ledger.measured_hot_share()
    low = P3_HOT_SHARE_TARGET - P3_HOT_SHARE_TOLERANCE
    high = P3_HOT_SHARE_TARGET + P3_HOT_SHARE_TOLERANCE
    if accounting.submitted >= 30 and not (low <= share <= high):
        raise AssertionError(
            f"P3 hot share out of envelope: measured={share:.4f} "
            f"target={P3_HOT_SHARE_TARGET} tol=±{P3_HOT_SHARE_TOLERANCE}",
        )
    accounting.assert_acceptance_reconciled()
    for cold in cold_tenants:
        if accounting.completions_by_tenant.get(cold, 0) <= 0:
            raise AssertionError(f"cold tenant {cold} made no completion progress")
    assert_cross_tenant_queue_fail_closed(
        queue,
        owner_tenant=hot_tenant,
        victim_tenant=cold_tenants[0],
    )
    return accounting, hot_ledger
