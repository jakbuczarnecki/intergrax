# © Artur Czarnecki. All rights reserved.

"""PLATFORM-5D — canonical root ExecutionId binding in NexusLoop.handle_task."""

from __future__ import annotations

import pytest

from intergrax.applications._shared.scenario_runtime_baseline import (
    ScenarioExecutionRequest,
    build_scenario_runtime_from_environment,
    execute_scenario_task,
)
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_id,
    peek_active_execution_identity,
    peek_active_parent_execution_id,
    require_active_execution_id,
    require_active_execution_identity,
    reset_active_execution_identity,
    validate_execution_id,
)
from intergrax.runtime.execution.boundary import ExecutionBoundary, ExecutionIdentityBinding
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.child import ChildExecutionRunner
from intergrax.runtime.execution.active_execution_budget import (
    bind_root_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget

_UNLIMITED_LEDGER = create_execution_budget_ledger(RunBudget())
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from echo.echo_agent import EchoAgent

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _fake_impl_factory(
    captured: dict[str, RunId | AttemptId | ExecutionId | None],
) -> object:
    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        captured["execution_id"] = require_active_execution_id()
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    return _fake_impl


@pytest.mark.asyncio
async def test_handle_task_mints_root_execution_id_without_prior_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()
    captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}
    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl_factory(captured))
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="root")

    await loop.handle_task(task, run_id=run_id)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] is not None
    assert validate_execution_id(captured["execution_id"])
    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_handle_task_supplied_run_id_path_has_root_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}
    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl_factory(captured))
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")

    await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id
    assert validate_execution_id(captured["execution_id"])


@pytest.mark.asyncio
async def test_handle_task_preserves_prebound_root_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    captured: dict[str, RunId | AttemptId | ExecutionId | None] = {}
    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl_factory(captured))
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="boundary")
    ledger = create_execution_budget_ledger(RunBudget())
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    budget_token = bind_root_execution_budget(
        execution_id=execution_id,
        ledger=ledger,
    )
    try:
        await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_budget(budget_token)
        reset_active_execution_identity(identity_token)

    assert captured["execution_id"] == execution_id


@pytest.mark.asyncio
async def test_handle_task_resets_identity_after_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()
    monkeypatch.setattr(
        loop,
        "_handle_task_impl",
        _fake_impl_factory({}),
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="ok")

    await loop.handle_task(task, run_id=run_id)

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_handle_task_resets_identity_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    run_id = mint_run_id()

    async def _boom(task: Task) -> TaskResult:
        require_active_execution_id()
        raise RuntimeError("boom")

    monkeypatch.setattr(loop, "_handle_task_impl", _boom)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="fail")

    with pytest.raises(RuntimeError, match="boom"):
        await loop.handle_task(task, run_id=run_id)

    assert peek_active_execution_identity() is None
    assert peek_active_execution_id() is None


@pytest.mark.asyncio
async def test_sequential_handle_task_invocations_do_not_share_execution_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = NexusLoop(AgentRegistry())
    seen: list[ExecutionId] = []

    async def _capture(task: Task) -> TaskResult:
        run_id, _ = require_active_execution_identity()
        seen.append(require_active_execution_id())
        return TaskResult(task_id=task.task_id, run_id=run_id, state=TaskState.COMPLETED)

    monkeypatch.setattr(loop, "_handle_task_impl", _capture)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="seq")

    await loop.handle_task(task, run_id=mint_run_id())
    await loop.handle_task(task, run_id=mint_run_id())

    assert len(seen) == 2
    assert seen[0] != seen[1]


@pytest.mark.asyncio
async def test_child_execution_parent_execution_id_points_to_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class Ping:
        value: str

    @dataclass(frozen=True)
    class Pong:
        value: str

    root_execution_id = mint_execution_id()
    child_captured: dict[str, ExecutionId | None] = {}
    child_runner = ChildExecutionRunner[Ping, Pong](ledger=_UNLIMITED_LEDGER)

    class ChildDelegate:
        async def execute(self, request: Ping) -> Pong:
            child_captured["execution_id"] = require_active_execution_id()
            child_captured["parent_execution_id"] = peek_active_parent_execution_id()
            return Pong(value=request.value)

    class RootDelegate:
        async def execute(self, request: Ping) -> Pong:
            assert require_active_execution_id() == root_execution_id
            await child_runner.execute(request=request, delegate=ChildDelegate())
            assert require_active_execution_id() == root_execution_id
            return Pong(value="ok")

    identity = ExecutionIdentityBinding(
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=root_execution_id,
    )
    await ExecutionBoundary[Ping, Pong](
        RootDelegate(),
        identity=identity,
        authority=ParentExecutionAuthority.unrestricted_root(),
    ).execute(
        Ping(value="child"),
    )

    assert child_captured["parent_execution_id"] == root_execution_id
    assert child_captured["execution_id"] != root_execution_id


@pytest.mark.asyncio
async def test_execute_scenario_task_reaches_graph_executor_with_execution_id(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path
    from testing_support.builder import MeteringFakeLLMAdapter

    adapter = MeteringFakeLLMAdapter()

    def _resolve(env: object, agent_override: object | None = None, **_: object) -> object:
        del env
        return agent_override or adapter

    monkeypatch.setattr(
        "intergrax.applications._shared.llm_resolver.resolve_llm_adapter",
        _resolve,
    )

    from intergrax.applications.contracts.environment_profile import (
        ApplicationEnvironmentProfile,
    )

    registry = AgentRegistry()
    registry.register(EchoAgent())
    environment = ApplicationEnvironmentProfile.lab_defaults(profile_id="scenario.platform.5d")
    manifest = ApplicationManifest.lab(
        app_id="scenario_platform_5d",
        name="Scenario Platform 5D",
        route_prefix="/v1/scenario_platform_5d",
        env_prefix="SCENARIO_PLATFORM_5D_",
        agents=[AgentBinding.mount(EchoAgent, capabilities=["echo.basic"])],
        environment=environment,
    )
    path = Path(str(tmp_path))
    composition = build_scenario_runtime_from_environment(
        environment=environment,
        registry=registry,
        tenant_id="scenario-platform-5d",
        manifest=manifest,
        runtime_events_db_path=path / "events.db",
        trace_db_path=path / "trace.db",
        use_in_memory_trace=True,
    )
    seen: dict[str, ExecutionId | None] = {}
    original_execute = composition.nexus_loop._graph_executor.execute

    async def _spy_execute(*args: object, **kwargs: object) -> object:
        seen["execution_id"] = peek_active_execution_id()
        return await original_execute(*args, **kwargs)

    composition.nexus_loop._graph_executor.execute = _spy_execute  # type: ignore[method-assign]

    result = await execute_scenario_task(
        composition,
        ScenarioExecutionRequest(
            tenant_id="scenario-platform-5d",
            message="hello",
            capability="echo.basic",
            task_id=mint_task_id(),
        ),
    )

    assert validate_execution_id(seen["execution_id"])
    assert result.task_result.state == TaskState.COMPLETED
