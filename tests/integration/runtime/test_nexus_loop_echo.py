# © Artur Czarnecki. All rights reserved.

import pytest

from echo.echo_agent import EchoAgent
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
async def test_nexus_loop_runs_echo_agent():
    registry = AgentRegistry()
    agent = EchoAgent()
    registry.register(agent)

    loop = NexusLoop(registry)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="hello harness",
        context=TaskContext(capability="echo.basic"),
    )

    result = await loop.handle_task(task)

    assert result.state == TaskState.COMPLETED
    assert "hello harness" in result.answer
    assert result.agent_id == "echo"
