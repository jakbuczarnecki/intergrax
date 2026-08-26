# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import ast
from dataclasses import dataclass, fields
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import mint_task_id
from intergrax.runtime.execution import ExecutionCapability, ExecutionRequest
from intergrax.runtime.execution.task_adapter import (
    TaskExecutionInput,
    execution_request_from_task,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskOrchestrationState, TaskRuntimeState

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_INPUT_FIELD_NAMES = frozenset(
    {
        "task_id",
        "run_id",
        "attempt_id",
        "execution_id",
        "tenant_id",
        "user_id",
        "session_id",
        "agent_id",
        "metadata",
        "canonical_identity",
        "execution_authority",
        "state",
        "options",
        "runtime",
    }
)

_FORBIDDEN_ADAPTER_NAMES = frozenset(
    {
        "Any",
        "Dict",
        "Mapping",
        "MutableMapping",
        "getattr",
        "setattr",
        "hasattr",
        "vars",
        "inspect",
        "importlib",
    }
)


@dataclass(frozen=True, slots=True)
class SummaryOutput:
    summary: str


def _minimal_task(**overrides: object) -> Task:
    base = {
        "task_id": mint_task_id(),
        "tenant_id": "tenant-1",
        "user_id": "user-1",
        "message": "hello",
        "context": TaskContext(capability="echo.basic", intent="greet"),
    }
    base.update(overrides)
    return Task(**base)  # type: ignore[arg-type]


def _agentic_task() -> Task:
    return Task(
        task_id=mint_task_id(),
        tenant_id="tenant-1",
        user_id="user-1",
        session_id="sess-1",
        agent_id="some_agent",
        message="use tools to investigate",
        context=TaskContext(
            capability="tool_using_incident_investigation",
            intent="investigate",
            metadata={"tool_hint": True},
        ),
        runtime=TaskRuntimeState(
            orchestration=TaskOrchestrationState(
                plan_id="plan-123",
                graph_id="graph-456",
            )
        ),
        metadata={"trace_id": "trace-abc"},
        state=TaskState.RUNNING,
    )


def test_message_maps_to_task_execution_input() -> None:
    task = _minimal_task(message="do the work")

    request = execution_request_from_task(task)

    assert request.input.message == "do the work"


def test_context_capability_maps_to_task_execution_input() -> None:
    task = _minimal_task(context=TaskContext(capability="incident.investigation"))

    request = execution_request_from_task(task)

    assert request.input.capability == "incident.investigation"


def test_context_intent_maps_to_task_execution_input() -> None:
    task = _minimal_task(context=TaskContext(intent="investigate"))

    request = execution_request_from_task(task)

    assert request.input.intent == "investigate"


def test_empty_context_values_remain_none() -> None:
    task = _minimal_task(context=TaskContext())

    request = execution_request_from_task(task)

    assert request.input.capability is None
    assert request.input.intent is None


def test_adapter_result_is_canonical_execution_request() -> None:
    request = execution_request_from_task(_minimal_task())

    assert isinstance(request, ExecutionRequest)


def test_request_input_is_task_execution_input_not_task() -> None:
    task = _minimal_task()

    request = execution_request_from_task(task)

    assert isinstance(request.input, TaskExecutionInput)
    assert not isinstance(request.input, Task)


def test_explicit_execution_capabilities_preserved_exactly() -> None:
    task = _agentic_task()
    capabilities = frozenset(
        {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
    )

    request = execution_request_from_task(task, capabilities=capabilities)

    assert request.capabilities == capabilities
    assert request.capabilities is capabilities


def test_default_execution_capabilities_are_empty() -> None:
    request = execution_request_from_task(_minimal_task())

    assert request.capabilities == frozenset()


def test_agent_id_does_not_imply_tools() -> None:
    request = execution_request_from_task(_agentic_task())

    assert request.capabilities == frozenset()


def test_runtime_orchestration_does_not_imply_orchestration() -> None:
    request = execution_request_from_task(_agentic_task())

    assert ExecutionCapability.ORCHESTRATION not in request.capabilities


def test_context_capability_string_does_not_imply_execution_capability() -> None:
    request = execution_request_from_task(_agentic_task())

    assert request.capabilities == frozenset()


def test_explicit_output_type_preserved_exactly() -> None:
    request = execution_request_from_task(
        _minimal_task(),
        output_type=SummaryOutput,
    )

    assert request.output_type is SummaryOutput


def test_no_output_type_supplied_defaults_to_none() -> None:
    request = execution_request_from_task(_minimal_task())

    assert request.output_type is None


def test_adapter_does_not_mutate_task() -> None:
    task = _agentic_task()
    before = task.model_dump()

    execution_request_from_task(task)

    assert task.model_dump() == before


def test_task_metadata_not_copied_into_projection() -> None:
    task = _agentic_task()

    request = execution_request_from_task(task)
    field_names = {field.name for field in fields(request.input)}

    assert "metadata" not in field_names
    assert field_names == {"message", "capability", "intent"}


def test_task_execution_input_excludes_identity_and_authority_fields() -> None:
    field_names = {field.name for field in fields(TaskExecutionInput)}

    assert field_names == {"message", "capability", "intent"}
    assert field_names.isdisjoint(_FORBIDDEN_INPUT_FIELD_NAMES)


def test_agent_id_not_field_of_task_execution_input() -> None:
    field_names = {field.name for field in fields(TaskExecutionInput)}

    assert "agent_id" not in field_names


def test_package_root_does_not_export_task_adapter_symbols() -> None:
    import intergrax.runtime.execution as execution_pkg

    assert "TaskExecutionInput" not in execution_pkg.__all__
    assert "execution_request_from_task" not in execution_pkg.__all__


def test_agentic_task_without_capabilities_stays_empty() -> None:
    request = execution_request_from_task(_agentic_task())

    assert request.capabilities == frozenset()


def test_explicit_capabilities_pass_through_without_inference() -> None:
    task = _agentic_task()
    capabilities = frozenset(
        {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
    )

    request = execution_request_from_task(task, capabilities=capabilities)

    assert request.capabilities == frozenset(
        {ExecutionCapability.TOOLS, ExecutionCapability.ORCHESTRATION}
    )


def test_task_adapter_module_avoids_dynamic_mechanisms() -> None:
    source = Path("intergrax/runtime/execution/task_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
    }

    assert names.isdisjoint(_FORBIDDEN_ADAPTER_NAMES)
