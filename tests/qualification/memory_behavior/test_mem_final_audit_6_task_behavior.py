# © Artur Czarnecki. All rights reserved.

"""MEM-FINAL-AUDIT-6 — TASK scope via MemoryControlPlane."""

from __future__ import annotations

import pytest

from tests.qualification.memory_behavior.contracts import BehaviorEvalContext
from tests.qualification.memory_behavior.scenarios.task import (
    run_task_01_remember_and_capability_read,
    run_task_02_forget,
    run_task_03_namespace_isolation,
    run_task_05_tenant_scope_on_task,
)

pytestmark = pytest.mark.gate


@pytest.mark.asyncio
async def test_task_01_remember_and_capability_read() -> None:
    await run_task_01_remember_and_capability_read(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_task_02_forget() -> None:
    await run_task_02_forget(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_task_03_namespace_isolation() -> None:
    await run_task_03_namespace_isolation(BehaviorEvalContext())


@pytest.mark.asyncio
async def test_task_05_tenant_scope_on_task() -> None:
    await run_task_05_tenant_scope_on_task(BehaviorEvalContext())
