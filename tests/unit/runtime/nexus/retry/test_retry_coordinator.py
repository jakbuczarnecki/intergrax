# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode
from intergrax.runtime.nexus.retry.coordinator import RetryCoordinator
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task, TaskContext

pytestmark = pytest.mark.gate


def test_should_retry_run_respects_policy() -> None:
    coordinator = RetryCoordinator(
        max_run_retries=2,
        retry_run_on=frozenset({RuntimeErrorCode.LLM_ERROR}),
    )
    assert coordinator.should_retry_run(
        attempt=0,
        error_code=RuntimeErrorCode.LLM_ERROR,
    )
    assert not coordinator.should_retry_run(
        attempt=2,
        error_code=RuntimeErrorCode.LLM_ERROR,
    )
    assert not coordinator.should_retry_run(
        attempt=0,
        error_code=RuntimeErrorCode.TOOL_ERROR,
    )


def test_scheduled_event_for_agent_retry() -> None:
    task = Task(tenant_id="t", user_id="u", context=TaskContext())
    coordinator = RetryCoordinator(max_run_retries=0, retry_run_on=frozenset())
    event = coordinator.scheduled_event_for_agent_retry(
        task,
        run_id="run-1",
        record=RetryRecord(
            attempt=1,
            agent_id="a1",
            reason="validation_failed",
            alternate_agent_id="a2",
        ),
    )
    assert event.event_type == RuntimeEventType.RETRY_SCHEDULED
    assert event.payload["alternate_agent_id"] == "a2"
