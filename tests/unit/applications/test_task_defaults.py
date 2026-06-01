# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Unit tests for harness task default injection (Phase M.11)."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.task_defaults import (
    apply_default_long_running_notify_channel,
    make_lab_harness_task_enricher,
)
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = pytest.mark.unit


def test_apply_default_notify_channel_when_long_running_enabled() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="x",
        context=TaskContext(capability="hitl.lr"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True, checkpoint_on_pause=True),
        ),
    )
    enriched = apply_default_long_running_notify_channel(task, default_channel="pagerduty")
    assert enriched.options.long_running.notify_channel == "pagerduty"


def test_apply_default_skips_when_channel_already_set() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="x",
        context=TaskContext(capability="hitl.lr"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(
                enabled=True,
                notify_channel="slack",
            ),
        ),
    )
    enriched = apply_default_long_running_notify_channel(task, default_channel="pagerduty")
    assert enriched.options.long_running.notify_channel == "slack"


def test_apply_default_skips_when_not_long_running() -> None:
    task = Task(tenant_id="t1", user_id="u1", message="x", context=TaskContext())
    enriched = apply_default_long_running_notify_channel(task, default_channel="pagerduty")
    assert enriched.options.long_running.notify_channel is None


def test_make_lab_harness_task_enricher_returns_none_for_non_harness() -> None:
    assert make_lab_harness_task_enricher(default_notify_channel="pagerduty", harness=False) is None


def test_make_lab_harness_task_enricher_applies_pagerduty() -> None:
    enricher = make_lab_harness_task_enricher(default_notify_channel="pagerduty", harness=True)
    assert enricher is not None
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="x",
        context=TaskContext(capability="hitl.lr"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    assert enricher(task).options.long_running.notify_channel == "pagerduty"
