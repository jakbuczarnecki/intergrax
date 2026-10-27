# © Artur Czarnecki. All rights reserved.

"""Dimension C overload step — rate-controlled offered load vs N=2 baseline."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskRequest, TaskStatus
from intergrax.queueing.providers.document_store import DocumentStoreTaskQueue

from tests.qualification.dist_scale_x.p2_task_load import (
    P2_TASK_PAYLOAD_BYTES,
    pending_task_count,
)

OVERLOAD_OFFER_MULTIPLIER = 1.5
OVERLOAD_RATE_TOLERANCE = 0.12
OVERLOAD_TICK_SECONDS = 0.1


@dataclass(slots=True)
class OverloadStepAccounting:
    sustainable_completed_per_sec: float
    target_offered_per_sec: float
    actual_offered_per_sec: float
    submitted: int = 0
    completed_terminal: int = 0
    failed_terminal: int = 0
    outstanding_after_step: int = 0
    max_queue_depth: int = 0
    depth_samples: list[int] = field(default_factory=list)
    depth_before_drain: int = 0
    depth_after_drain: int = 0
    handles: list[TaskHandle] = field(default_factory=list)


def run_rate_controlled_overload_step(
    queue: DocumentStoreTaskQueue,
    *,
    sustainable_completed_per_sec: float,
    duration_s: float,
    task_name: str,
    tenant_id: str = "tenant-a",
) -> OverloadStepAccounting:
    if sustainable_completed_per_sec <= 0:
        raise ValueError("sustainable_completed_per_sec must be positive")
    target_rate = sustainable_completed_per_sec * OVERLOAD_OFFER_MULTIPLIER
    accounting = OverloadStepAccounting(
        sustainable_completed_per_sec=sustainable_completed_per_sec,
        target_offered_per_sec=target_rate,
        actual_offered_per_sec=0.0,
    )
    target_total = max(1, int(round(target_rate * duration_s)))
    tick_budget = max(1, int(round(duration_s / OVERLOAD_TICK_SECONDS)))
    base_per_tick = target_total // tick_budget
    extra_ticks = target_total % tick_budget
    deadline = time.monotonic() + duration_s
    tick_index = 0
    while time.monotonic() < deadline:
        batch = base_per_tick + (1 if tick_index < extra_ticks else 0)
        tick_index += 1
        for _ in range(batch):
            handle = queue.enqueue(
                TaskRequest(
                    tenant_id=tenant_id,
                    run_id=f"burst-{accounting.submitted}",
                    task_name=task_name,
                    payload=b"x" * P2_TASK_PAYLOAD_BYTES,
                )
            )
            accounting.handles.append(handle)
            accounting.submitted += 1
        depth = pending_task_count(queue, tenant_id)
        accounting.depth_samples.append(depth)
        accounting.max_queue_depth = max(accounting.max_queue_depth, depth)
        time.sleep(OVERLOAD_TICK_SECONDS)
    accounting.actual_offered_per_sec = accounting.submitted / float(duration_s)
    for handle in accounting.handles:
        status = queue.get_status(handle)
        if status is TaskStatus.SUCCEEDED:
            accounting.completed_terminal += 1
        elif status is TaskStatus.FAILED:
            accounting.failed_terminal += 1
    accounting.outstanding_after_step = (
        accounting.submitted - accounting.completed_terminal - accounting.failed_terminal
    )
    accounting.depth_before_drain = pending_task_count(queue, tenant_id)
    return accounting


def assert_overload_acceptance(accounting: OverloadStepAccounting) -> None:
    low = accounting.target_offered_per_sec * (1.0 - OVERLOAD_RATE_TOLERANCE)
    high = accounting.target_offered_per_sec * (1.0 + OVERLOAD_RATE_TOLERANCE)
    if not (low <= accounting.actual_offered_per_sec <= high):
        raise AssertionError(
            "offered rate outside 1.5x tolerance: "
            f"target={accounting.target_offered_per_sec:.3f}/s "
            f"actual={accounting.actual_offered_per_sec:.3f}/s "
            f"band=[{low:.3f},{high:.3f}]",
        )
    if accounting.submitted <= 0:
        raise AssertionError("no tasks offered during overload step")
    backlog_observed = (
        accounting.max_queue_depth > 0
        or accounting.outstanding_after_step > 0
        or accounting.submitted > accounting.completed_terminal
    )
    if not backlog_observed:
        raise AssertionError(
            "required overload did not produce backlog or incomplete work — queue behavior not confirmed",
        )
