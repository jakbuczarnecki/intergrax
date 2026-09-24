# © Artur Czarnecki. All rights reserved.

"""EBH-2F-R1 — Tier-3 serving consumes HostTaskExecutionPort without Nexus."""

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.execution.host_task import HostTaskExecutionPort
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from local_workspace_application.host.lifecycle import LocalWorkspaceHostLifecycle
from local_workspace_application.host.task_executor import LocalWorkspaceTaskExecutor

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _FakeHostExecution:
    def __init__(self) -> None:
        self.calls = 0

    async def execute(
        self,
        task: Task,
        *,
        run_id=None,
        attempt_id=None,
        execution_id=None,
        resume_checkpoint=None,
        restore_existing_execution: bool = False,
    ) -> TaskResult:
        self.calls += 1
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id or mint_run_id(),
            state=TaskState.COMPLETED,
            answer="fake",
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )


@pytest.mark.asyncio
async def test_lkw_task_executor_accepts_custom_host_execution_port_without_nexus() -> None:
    fake = _FakeHostExecution()
    lifecycle = LocalWorkspaceHostLifecycle()
    lifecycle.transition_to_ready()
    lifecycle.set_executor_available(True)
    executor = LocalWorkspaceTaskExecutor(
        fake,
        task_enricher=None,
        readiness=lifecycle,
    )
    task = Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        message="replaceability",
        context=TaskContext(capability="local.workspace.search"),
    )

    await executor.execute(task)
    assert fake.calls == 1


def test_fake_host_execution_satisfies_port_protocol() -> None:
    fake: HostTaskExecutionPort = _FakeHostExecution()
    assert callable(getattr(fake, "execute", None))
