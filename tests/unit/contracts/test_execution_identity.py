# © Artur Czarnecki. All rights reserved.

import asyncio
import re

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    require_active_execution_identity,
    reset_active_execution_identity,
    transition_active_execution_identity,
    validate_attempt_id,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.task.task import Task
from intergrax.runtime.task.task_run_bridge import (
    new_run_id,
    task_from_runtime_request,
)
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner

_CANONICAL_ID = re.compile(r"^(task|run|attempt|evt)_[0-9a-f]{32}$")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_task_id_accepts_valid():
    value = mint_task_id()
    assert validate_task_id(value) == value
    assert _CANONICAL_ID.fullmatch(validate_task_id(value))


@pytest.mark.unit
@pytest.mark.gate
def test_validate_run_id_accepts_valid():
    value = mint_run_id()
    assert validate_run_id(value) == value


@pytest.mark.unit
@pytest.mark.gate
def test_validate_attempt_id_accepts_valid():
    value = mint_attempt_id()
    assert validate_attempt_id(value) == value


@pytest.mark.unit
@pytest.mark.gate
def test_validate_rejects_wrong_prefix():
    task_id = mint_task_id()
    with pytest.raises(ValueError, match="TaskId must start with"):
        validate_task_id(task_id.replace("task_", "run_", 1))


@pytest.mark.unit
@pytest.mark.gate
def test_validate_rejects_uppercase_hex():
    task_id = mint_task_id()
    upper = "task_" + task_id.split("_", 1)[1].upper()
    with pytest.raises(ValueError, match="suffix"):
        validate_task_id(upper)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_rejects_malformed_suffix():
    with pytest.raises(ValueError, match="suffix"):
        validate_run_id("run_tooshort")


@pytest.mark.unit
@pytest.mark.gate
def test_validate_rejects_non_string():
    with pytest.raises(TypeError):
        validate_task_id(123)
    with pytest.raises(TypeError):
        validate_run_id(object())


@pytest.mark.unit
@pytest.mark.gate
def test_validate_rejects_whitespace():
    with pytest.raises(ValueError):
        validate_task_id("   ")
    with pytest.raises(ValueError):
        validate_run_id(" run_0123456789abcdef0123456789abcdef")


@pytest.mark.unit
@pytest.mark.gate
def test_task_creation_mints_task_id():
    task = Task(tenant_id="t1", user_id="u1", message="hello")
    assert _CANONICAL_ID.fullmatch(task.task_id)


@pytest.mark.unit
@pytest.mark.gate
def test_new_run_id_mints_independent_run_id():
    run_id = new_run_id()
    assert _CANONICAL_ID.fullmatch(run_id)
    assert run_id.startswith("run_")


@pytest.mark.unit
@pytest.mark.gate
def test_task_to_runtime_request_requires_explicit_run_id():
    task = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="agent-1",
        message="execute",
    )
    run_id = mint_run_id()
    request = task.to_runtime_request(run_id=run_id)
    assert request.task_id == task.task_id
    assert request.run_id == run_id
    assert request.run_id != task.task_id
    assert request.metadata["run_id"] == run_id
    assert request.metadata["task_id"] == task.task_id


@pytest.mark.unit
@pytest.mark.gate
def test_task_from_runtime_request_uses_request_task_id_not_run_id():
    task_id = mint_task_id()
    run_id = mint_run_id()
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="u1",
        session_id="sess_0123456789abcdef0123456789abcdef",
        message="hello",
        task_id=task_id,
        run_id=run_id,
        tenant_id="t1",
        metadata={"run_id": "run_legacyshouldnotapply0123456789abcdef"},
    )
    task = task_from_runtime_request(request, tenant_id="t1", user_id="u1")
    assert task.task_id == task_id
    assert task.task_id != run_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unified_task_runner_mints_attempt_at_run_boundary():
    minted_attempt: AttemptId | None = None

    class _StubLoop:
        def __init__(self) -> None:
            from intergrax.runtime.nexus.execution.graph_executor import GraphExecutor
            from intergrax.runtime.registry.agent_registry import AgentRegistry

            registry = AgentRegistry()
            self._graph_executor = GraphExecutor(registry)

        async def handle_task(
            self,
            task: Task,
            *,
            run_id: RunId,
            attempt_id: AttemptId | None = None,
        ):
            nonlocal minted_attempt
            resolved_attempt_id = attempt_id or mint_attempt_id()
            bind_active_execution_identity(
                run_id=run_id,
                attempt_id=resolved_attempt_id,
            )
            minted_attempt = resolved_attempt_id
            from intergrax.runtime.task.task import TaskResult, TaskState

            return TaskResult(
                task_id=task.task_id,
                run_id=run_id,
                state=TaskState.COMPLETED,
            )

    loop = _StubLoop()
    runner = UnifiedTaskRunner(loop)
    task = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="agent-1",
        message="attempt boundary",
    )
    result = await runner.run_task(task)
    assert minted_attempt is not None
    assert _CANONICAL_ID.fullmatch(minted_attempt)
    assert result.run_id != task.task_id


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_request_requires_task_and_run_id_fields():
    request = RuntimeRequest(
        agent_id="agent-1",
        user_id="u1",
        session_id="sess_0123456789abcdef0123456789abcdef",
        message="hello",
        task_id=mint_task_id(),
        run_id=mint_run_id(),
    )
    assert _CANONICAL_ID.fullmatch(request.task_id)
    assert _CANONICAL_ID.fullmatch(request.run_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_handle_task_initial_execution_mints_attempt_id(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.task import TaskResult, TaskState

    captured: dict[str, RunId | AttemptId] = {}
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()

    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
        )

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="execute")
    await loop.handle_task(task, run_id=run_id)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] is not None
    assert _CANONICAL_ID.fullmatch(captured["attempt_id"])
    assert captured["run_id"] != task.task_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_handle_task_resume_preserves_run_and_attempt_id(monkeypatch):
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.task import TaskResult, TaskState

    captured: dict[str, RunId | AttemptId] = {}
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    async def _fake_impl(task: Task) -> TaskResult:
        run_id, attempt_id = require_active_execution_identity()
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
        )

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_handle_task_resume_does_not_mint_attempt_id(monkeypatch):
    from intergrax.runtime.nexus import nexus_loop as nexus_loop_module
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.task import TaskResult, TaskState

    def _forbidden_mint() -> AttemptId:
        raise AssertionError("mint_attempt_id must not be called on resume")

    monkeypatch.setattr(nexus_loop_module, "mint_attempt_id", _forbidden_mint)

    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    async def _fake_impl(task: Task) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
        )

    monkeypatch.setattr(loop, "_handle_task_impl", _fake_impl)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)


@pytest.mark.unit
@pytest.mark.gate
def test_multi_agent_evaluation_requires_active_run_id():
    from intergrax.contracts.agent_execution_result import (
        AgentExecutionResult,
        AgentExecutionStatus,
    )
    from intergrax.runtime.architecture.online_evaluation_registry import (
        InMemoryOnlineEvaluationRegistry,
    )
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry

    loop = NexusLoop(AgentRegistry())
    loop._evaluation_registry = InMemoryOnlineEvaluationRegistry()
    execution = AgentExecutionResult(
        agent_id="a1",
        run_id=mint_run_id(),
        status=AgentExecutionStatus.COMPLETED,
        summary="ok",
    )

    with pytest.raises(RuntimeError, match="active execution identity required"):
        loop._maybe_record_multi_agent_evaluation(
            [execution, execution],
            task_id="task_0123456789abcdef0123456789abcdef",
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_unified_task_runner_resume_uses_checkpoint_identity(monkeypatch):
    from intergrax.runtime.long_running.models import TaskCheckpoint
    from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
    from intergrax.runtime.nexus.nexus_loop import NexusLoop
    from intergrax.runtime.registry.agent_registry import AgentRegistry
    from intergrax.runtime.task.task import TaskResult, TaskState

    captured: dict[str, RunId | AttemptId | None] = {}
    registry = AgentRegistry()
    loop = NexusLoop(registry)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()

    async def _fake_handle_task(
        task: Task,
        *,
        run_id: RunId,
        attempt_id: AttemptId | None = None,
    ) -> TaskResult:
        captured["run_id"] = run_id
        captured["attempt_id"] = attempt_id
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
        )

    monkeypatch.setattr(loop, "handle_task", _fake_handle_task)
    runner = UnifiedTaskRunner(loop)
    task = Task(tenant_id="t1", user_id="u1", agent_id="agent-1", message="resume")
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        runtime=RuntimeCheckpoint(run_id=run_id, attempt_id=attempt_id),
    )

    await runner.run_task(task, resume_checkpoint=checkpoint)

    assert captured["run_id"] == run_id
    assert captured["attempt_id"] == attempt_id


@pytest.mark.unit
@pytest.mark.gate
def test_bind_require_reset_execution_identity() -> None:
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_id)
    assert require_active_execution_identity() == (run_id, attempt_id)
    reset_active_execution_identity(token)
    with pytest.raises(RuntimeError, match="active execution identity required"):
        require_active_execution_identity()


@pytest.mark.unit
@pytest.mark.gate
def test_nested_execution_identity_binding() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_a2 = mint_attempt_id()
    outer_token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1)
    assert require_active_execution_identity() == (run_id, attempt_a1)
    inner_token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a2)
    assert require_active_execution_identity() == (run_id, attempt_a2)
    reset_active_execution_identity(inner_token)
    assert require_active_execution_identity() == (run_id, attempt_a1)
    reset_active_execution_identity(outer_token)
    assert peek_active_execution_identity() is None


@pytest.mark.unit
@pytest.mark.gate
def test_transition_retry_preserves_run_id() -> None:
    run_id = mint_run_id()
    attempt_a1 = mint_attempt_id()
    token = bind_active_execution_identity(run_id=run_id, attempt_id=attempt_a1)
    attempt_a2 = transition_active_execution_identity()
    assert attempt_a2 != attempt_a1
    bound_run_id, bound_attempt_id = require_active_execution_identity()
    assert bound_run_id == run_id
    assert bound_attempt_id == attempt_a2
    reset_active_execution_identity(token)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_concurrent_execution_identity_isolation() -> None:
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_b1 = mint_attempt_id()
    gate = asyncio.Event()
    results: dict[str, tuple[RunId, AttemptId]] = {}

    async def coroutine_a() -> None:
        token = bind_active_execution_identity(run_id=run_r1, attempt_id=attempt_a1)
        try:
            gate.set()
            await asyncio.sleep(0.05)
            results["a"] = require_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    async def coroutine_b() -> None:
        await gate.wait()
        token = bind_active_execution_identity(run_id=run_r2, attempt_id=attempt_b1)
        try:
            await asyncio.sleep(0.05)
            results["b"] = require_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    await asyncio.gather(coroutine_a(), coroutine_b())
    assert results["a"] == (run_r1, attempt_a1)
    assert results["b"] == (run_r2, attempt_b1)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.asyncio
async def test_concurrent_retry_transition_isolation() -> None:
    run_r1 = mint_run_id()
    run_r2 = mint_run_id()
    attempt_a1 = mint_attempt_id()
    attempt_b1 = mint_attempt_id()
    gate = asyncio.Event()
    results: dict[str, AttemptId | tuple[RunId, AttemptId]] = {}

    async def coroutine_a() -> None:
        token = bind_active_execution_identity(run_id=run_r1, attempt_id=attempt_a1)
        try:
            gate.set()
            await asyncio.sleep(0.05)
            results["a"] = transition_active_execution_identity()
        finally:
            reset_active_execution_identity(token)

    async def coroutine_b() -> None:
        await gate.wait()
        token = bind_active_execution_identity(run_id=run_r2, attempt_id=attempt_b1)
        try:
            await asyncio.sleep(0.05)
            results["b_before"] = require_active_execution_identity()[1]
            await asyncio.sleep(0.05)
            results["b_after"] = require_active_execution_identity()[1]
        finally:
            reset_active_execution_identity(token)

    await asyncio.gather(coroutine_a(), coroutine_b())
    assert results["a"] != attempt_a1
    assert results["b_before"] == attempt_b1
    assert results["b_after"] == attempt_b1
