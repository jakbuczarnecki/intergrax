# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Harness M.11 — default notify_channel from lab wiring without explicit task field."""

from __future__ import annotations

from typing import Any

import pytest

from intergrax.applications._shared.task_defaults import make_lab_harness_task_enricher
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = pytest.mark.unit


def test_harness_enricher_injects_pagerduty_without_explicit_notify_channel() -> None:
    enricher = make_lab_harness_task_enricher(default_notify_channel="pagerduty", harness=True)
    assert enricher is not None
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="sensitive",
        context=TaskContext(capability="hitl.lr"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, checkpoint_on_pause=True),
        ),
    )
    assert task.options.long_running.notify_channel is None
    enriched = enricher(task)
    assert enriched.options.long_running.notify_channel == "pagerduty"
    assert enriched.metadata.get("long_running_notify_channel") == "pagerduty"
