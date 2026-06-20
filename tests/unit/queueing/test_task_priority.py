# © Artur Czarnecki. All rights reserved.

"""ORCH-MAINT-03 — TaskPriority on queue contracts."""

from __future__ import annotations

import pytest

from intergrax.queueing.contracts.task_queue import TaskPriority, TaskRequest
from intergrax.queueing.task_priority import TaskPriority as PriorityEnum

pytestmark = pytest.mark.unit


def test_task_request_default_priority() -> None:
    req = TaskRequest(
        tenant_id="t",
        run_id="r",
        task_name="demo",
        payload=b"{}",
    )
    assert req.priority is TaskPriority.NORMAL


def test_task_priority_coerce() -> None:
    assert PriorityEnum.coerce("HIGH") is PriorityEnum.HIGH
    assert PriorityEnum.coerce(0) is PriorityEnum.CRITICAL
