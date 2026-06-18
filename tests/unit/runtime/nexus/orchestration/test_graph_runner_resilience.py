# © Artur Czarnecki. All rights reserved.

"""Tests for graph_runner resilience policy wiring (FLOW-MAINT-01)."""

from __future__ import annotations

import pytest

from intergrax.contracts.resilience_policy import ResiliencePolicy
from intergrax.runtime.nexus.orchestration.graph_runner import NexusGraphRunner
from intergrax.runtime.nexus.retry.retry_engine import _resilience_policy_from_task
from intergrax.runtime.task.task import Task

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_resilience_policy_disallows_partial_result() -> None:
    task = Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        message="hello",
        metadata={
            "resilience_policy.v1": ResiliencePolicy(allow_partial_result=False).model_dump(),
        },
    )
    policy = _resilience_policy_from_task(task)
    assert policy is not None
    assert policy.allow_partial_result is False


def test_graph_runner_module_exports_runner() -> None:
    assert NexusGraphRunner is not None
