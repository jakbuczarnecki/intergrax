# © Artur Czarnecki. All rights reserved.

import pytest

from local_search.local_search_agent import LocalSearchAgent
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_local_search_agent_runs_through_nexus():
    registry = AgentRegistry()
    registry.register(LocalSearchAgent())
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="scaffold smoke",
            context=TaskContext(capability="local.workspace.search"),
        )
    )
    assert result.state == TaskState.COMPLETED
    assert result.agent_id == "local_search"
    assert "not_implemented" not in result.answer
