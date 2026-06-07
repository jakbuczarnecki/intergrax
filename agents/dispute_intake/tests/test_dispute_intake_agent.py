# © Artur Czarnecki. All rights reserved.

import pytest

from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_dispute_intake_agent_runs_through_nexus():
    registry = AgentRegistry()
    registry.register(DisputeIntakeAgent())
    loop = NexusLoop(registry)
    result = await loop.handle_task(
        Task(
            tenant_id="t1",
            user_id="u1",
            message="scaffold smoke",
            context=TaskContext(capability="dispute.intake"),
        )
    )
    assert result.state == TaskState.COMPLETED
    assert "scaffold smoke" in result.answer
    assert result.agent_id == "dispute_intake"
