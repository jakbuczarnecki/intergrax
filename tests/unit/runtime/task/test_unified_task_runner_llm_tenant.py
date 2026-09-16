# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock

import pytest

from intergrax.runtime.nexus.nexus_loop import NexusLoop

from intergrax.llm_adapters.tracking.context import get_llm_tenant_id
from intergrax.runtime.task.task import TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.runtime.task.task import TaskResult
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.builder import build_stub_nexus_loop_for_unified_task_runner, build_task_for_tests

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_unified_task_runner_sets_llm_tenant_scope() -> None:
    captured: list[str] = []

    async def _handle(task, *, run_id, attempt_id=None):
        captured.append(get_llm_tenant_id())
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    loop = cast(NexusLoop, build_stub_nexus_loop_for_unified_task_runner())
    loop.handle_task = _handle  # type: ignore[attr-defined]
    runner = UnifiedTaskRunner(loop)

    task = build_task_for_tests(
        seed="llm-tenant",
        tenant_id="tenant-42",
        user_id="user-1",
    )
    await runner.run_task(task)
    assert captured == ["tenant-42"]
