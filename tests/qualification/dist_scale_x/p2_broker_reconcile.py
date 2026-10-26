# © Artur Czarnecki. All rights reserved.

"""Broker terminal reconciliation via BrokerBackedTaskQueue KV contract."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.queueing.contracts.task_queue import TaskHandle, TaskQueue, TaskStatus


@dataclass(slots=True)
class BrokerTerminalAccounting:
    submitted: int
    succeeded: int
    failed: int
    pending: int

    @property
    def terminal(self) -> int:
        return self.succeeded + self.failed


def reconcile_broker_task_handles(
    queue: TaskQueue,
    handles: list[TaskHandle],
) -> BrokerTerminalAccounting:
    succeeded = 0
    failed = 0
    pending = 0
    seen: set[str] = set()
    for handle in handles:
        if handle.task_id in seen:
            raise AssertionError(f"duplicate task_id in submission ledger: {handle.task_id}")
        seen.add(handle.task_id)
        status = queue.get_status(handle)
        if status is TaskStatus.SUCCEEDED:
            result = queue.get_result(handle)
            if result is None or result.status is not TaskStatus.SUCCEEDED:
                raise AssertionError(
                    f"KV status SUCCEEDED without terminal result for {handle.task_id}",
                )
            succeeded += 1
        elif status is TaskStatus.FAILED:
            failed += 1
        else:
            pending += 1
    return BrokerTerminalAccounting(
        submitted=len(handles),
        succeeded=succeeded,
        failed=failed,
        pending=pending,
    )


def assert_broker_run_acceptance(accounting: BrokerTerminalAccounting) -> None:
    if accounting.submitted <= 0:
        raise AssertionError("no broker submissions recorded")
    if accounting.pending > 0:
        raise AssertionError(f"outstanding broker tasks after drain: {accounting.pending}")
    if accounting.succeeded != accounting.submitted:
        raise AssertionError(
            f"broker terminal mismatch: submitted={accounting.submitted} "
            f"succeeded={accounting.succeeded} failed={accounting.failed}",
        )
