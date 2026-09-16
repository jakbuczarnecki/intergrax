# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.orchestration.human_response import normalize_human_response
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskHumanInput
from testing_support.builder import build_task_for_tests

pytestmark = pytest.mark.gate


def _task_with_human_response(text: str) -> Task:
    return build_task_for_tests(
        seed="nexus-loop-human",
        tenant_id="tenant",
        user_id="user",
        agent_id="agent",
        message="q",
    ).model_copy(
        update={"options": TaskExecutionOptions(human=TaskHumanInput(response_text=text))}
    )


def test_normalize_human_response_records_verdict() -> None:
    task = _task_with_human_response("approved")
    normalize_human_response(task)
    assert task.options.human.verdict is not None


@pytest.mark.asyncio
async def test_nexus_loop_exposes_middleware() -> None:
    loop = NexusLoop(AgentRegistry())
    assert loop.middleware is not None


def test_nexus_loop_wires_intake_and_planning_runners() -> None:
    loop = NexusLoop(AgentRegistry())
    assert type(loop._intake_runner).__name__ == "NexusIntakeRunner"
    assert type(loop._planning_runner).__name__ == "NexusPlanningRunner"


def test_nexus_loop_policy_engine_is_facade() -> None:
    loop = NexusLoop(AgentRegistry())
    from intergrax.runtime.policy.policy_engine import PolicyEngine

    assert isinstance(loop.policy_engine, PolicyEngine)


def test_persist_human_decision_no_store() -> None:
    loop = NexusLoop(AgentRegistry())
    task = _task_with_human_response("")
    loop._persist_human_decision(task, HumanResponseVerdict.APPROVE)
