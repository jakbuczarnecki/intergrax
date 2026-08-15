# © Artur Czarnecki. All rights reserved.

import re

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    RunId,
    TaskId,
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
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

_CANONICAL_ID = re.compile(r"^(task|run|attempt)_[0-9a-f]{32}$")


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
    assert "run_id" not in request.metadata
    assert "task_id" not in request.metadata


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

        async def handle_task(self, task: Task, *, run_id: RunId):
            nonlocal minted_attempt
            attempt_id = mint_attempt_id()
            self._graph_executor.set_execution_identity(
                run_id=run_id,
                attempt_id=attempt_id,
            )
            minted_attempt = self._graph_executor.execution_attempt_id
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
