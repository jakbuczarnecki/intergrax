# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.applications._shared.async_task_dispatch import InMemoryAsyncTaskIndex, run_async
from intergrax.applications._shared.reliability_wiring import apply_reliability_task_defaults
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.contracts.agent_contract_meta import AgentExecutionMode
from intergrax.contracts.autonomy_level import AutonomyLevel
from intergrax.contracts.execution_identity import mint_run_id, mint_task_id
from intergrax.runtime.task.active_task_registry import ActiveTaskRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
async def test_run_async_returns_pending_handle() -> None:
    class _Runner:
        async def run_task(self, task: Task) -> TaskResult:
            return TaskResult(
                task_id=task.task_id,
                state=TaskState.COMPLETED,
                answer="ok",
            )

    index = InMemoryAsyncTaskIndex()
    payload = await run_async(
        _Runner(),
        Task(task_id="t1", tenant_id="t", user_id="u", message="hi"),
        index=index,
    )
    assert payload["status"] == "pending"
    assert payload["async"] is True


def test_async_batch_defaults_preset() -> None:
    env = ApplicationEnvironmentProfile.async_batch_defaults()
    assert env.orchestration_profile.long_running_enabled is True
    assert env.reliability_profile.long_running_scheduler_enabled is True


def test_apply_reliability_task_defaults_sets_autonomy() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults()
    env.reliability_profile.default_autonomy_level = AutonomyLevel.MANUAL
    task = apply_reliability_task_defaults(
        Task(
            task_id="t",
            tenant_id="t",
            user_id="u",
            message="m",
            context=TaskContext(capability="echo.basic"),
        ),
        env,
    )
    assert task.options.governance.autonomy_level is AutonomyLevel.MANUAL
    assert task.metadata["resilience_policy.v1"]["policy_id"] == "harness.default"


@pytest.mark.asyncio
async def test_active_task_registry_tracks_inflight_task() -> None:
    ActiveTaskRegistry.clear_for_tests()
    task_id = mint_task_id()
    run_id = mint_run_id()
    task = Task(task_id=task_id, tenant_id="t", user_id="u", message="m")
    await ActiveTaskRegistry.register(task, run_id)
    binding = await ActiveTaskRegistry.get(task_id)
    assert binding is not None
    assert binding.task is task
    assert binding.run_id == run_id
    await ActiveTaskRegistry.unregister(task_id, run_id)
    assert await ActiveTaskRegistry.get(task_id) is None


def test_agent_contract_default_execution_mode_is_async() -> None:
    from intergrax.contracts.agent_contract_meta import AgentContract

    contract = AgentContract(
        id="demo",
        name="Demo",
        description="d",
        version="1",
        capabilities=["echo.basic"],
        input_schema={},
        output_schema={},
    )
    assert contract.execution_mode is AgentExecutionMode.ASYNC
