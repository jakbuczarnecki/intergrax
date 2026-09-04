# © Artur Czarnecki. All rights reserved.

"""P2-003-D2: Task/events package facade import boundary regression gate."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]

_IMPORT_ISOLATION_CASES = (
    "from intergrax.runtime.task.task import Task, TaskState",
    "import intergrax.runtime.task as task; assert task.Task is not None",
    "from intergrax.runtime.task import TaskTraceEmitter; assert TaskTraceEmitter is not None",
    "import intergrax.runtime.events as events; assert events.RuntimeEvent is not None",
    "from intergrax.runtime.events import trace_event_to_runtime_event; assert trace_event_to_runtime_event is not None",
    "from intergrax.runtime.events.trace_bridge import trace_event_to_runtime_event; assert trace_event_to_runtime_event is not None",
    "import intergrax.integrations",
)


@pytest.mark.parametrize("statement", _IMPORT_ISOLATION_CASES)
def test_task_events_import_boundary_subprocess(statement: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_task_events_package_facade_lazy_exports() -> None:
    from intergrax.runtime.events import runtime_event_from_task_state, trace_event_to_runtime_event
    from intergrax.runtime.task import (
        PersistingTaskTraceEmitter,
        TaskTraceEmitter,
        lifecycle_with_persisting_trace,
        lifecycle_with_trace,
        new_run_id,
        task_from_execution_request,
        task_from_runtime_request,
        task_result_to_payload,
        task_to_execution_payload,
    )

    assert runtime_event_from_task_state is not None
    assert trace_event_to_runtime_event is not None
    assert TaskTraceEmitter is not None
    assert PersistingTaskTraceEmitter is not None
    assert lifecycle_with_trace is not None
    assert lifecycle_with_persisting_trace is not None
    assert new_run_id is not None
    assert task_from_execution_request is not None
    assert task_from_runtime_request is not None
    assert task_result_to_payload is not None
    assert task_to_execution_payload is not None
