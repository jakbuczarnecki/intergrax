# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.human_response import normalize_human_response
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskHumanInput

pytestmark = pytest.mark.gate


def _task_with_human_response(text: str) -> Task:
    return Task(
        task_id="t1",
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
        options=TaskExecutionOptions(human=TaskHumanInput(response_text=text)),
    )


def test_normalize_human_response_records_verdict() -> None:
    task = _task_with_human_response("approved")
    normalize_human_response(task)
    assert task.options.human.verdict is not None


@pytest.mark.asyncio
async def test_nexus_loop_exposes_middleware() -> None:
    loop = NexusLoop(AgentRegistry())
    assert loop.middleware is not None


def test_nexus_loop_policy_engine_is_facade() -> None:
    loop = NexusLoop(AgentRegistry())
    from intergrax.runtime.policy.policy_engine import PolicyEngine

    assert isinstance(loop.policy_engine, PolicyEngine)


def test_persist_human_decision_no_store() -> None:
    loop = NexusLoop(AgentRegistry())
    task = _task_with_human_response("")
    loop._persist_human_decision(task, HumanResponseVerdict.APPROVE)
