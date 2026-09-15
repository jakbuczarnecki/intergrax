# © Artur Czarnecki. All rights reserved.

"""R1 TaskResult authoritative exposure invariant tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.task.task import TaskResult
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task_state import TaskState

pytestmark = pytest.mark.unit


def test_r1_t3_non_terminal_waiting_human_without_exposure_valid() -> None:
    result = TaskResult(task_id="task-1", state=TaskState.WAITING_FOR_HUMAN)
    assert result.authoritative_decision_exposure is None


def test_r1_t4_explicit_no_decision_gate_valid() -> None:
    exposure = terminal_task_result_exposure_no_decision_gate()
    result = TaskResult(
        task_id="task-1",
        state=TaskState.COMPLETED,
        authoritative_decision_exposure=exposure,
    )
    assert result.authoritative_decision_exposure is exposure
