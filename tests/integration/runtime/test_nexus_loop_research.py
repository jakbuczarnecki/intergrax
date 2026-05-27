# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.bootstrap import build_research_registry
from intergrax.runtime.task.task import Task, TaskContext, TaskState


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.gate
async def test_nexus_loop_research_pipeline_via_uaep():
    loop = NexusLoop(build_research_registry())
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="competitors in renewable energy Poland",
        context=TaskContext(capability="research.pipeline", intent="research_summarize"),
    )

    result = await loop.handle_task(task)

    assert result.state in {TaskState.COMPLETED, TaskState.PARTIALLY_COMPLETED}
    assert result.answer
    assert result.metadata.get("validation_valid") is True
