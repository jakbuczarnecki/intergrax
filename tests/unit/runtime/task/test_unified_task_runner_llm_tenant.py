# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from intergrax.llm_adapters.tracking.context import get_llm_tenant_id
from intergrax.runtime.task.task import Task, TaskContext
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_unified_task_runner_sets_llm_tenant_scope() -> None:
    captured: list[str] = []

    async def _handle(task: Task):
        captured.append(get_llm_tenant_id())
        return MagicMock()

    loop = MagicMock()
    loop.handle_task = _handle
    runner = UnifiedTaskRunner(loop)  # type: ignore[arg-type]

    task = Task(
        tenant_id="tenant-42",
        user_id="user-1",
        context=TaskContext(),
    )
    await runner.run_task(task)
    assert captured == ["tenant-42"]
