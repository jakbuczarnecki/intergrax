# © Artur Czarnecki. All rights reserved.

"""OBS-RUNTIME-HISTORY-BOUNDS-R4: reflection-free metric scope propagation."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.execution_identity import (
    AttemptId,
    ExecutionId,
    RunId,
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    reset_active_execution_identity,
)
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event_metric_scope import RuntimeEventMetricScope
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority

_NEXUS_LOOP_PATH = Path("intergrax/runtime/nexus/nexus_loop.py")

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _bind_nexus_context(
    *,
    run_id: RunId,
    attempt_id: AttemptId,
    execution_id: ExecutionId,
):
    ledger = create_execution_budget_ledger(RunBudget(max_total_tokens=100))
    identity_token = bind_active_execution_identity(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    authority_token = bind_active_execution_authority(
        ParentExecutionAuthority.unrestricted_root(),
    )
    budget_token = bind_active_execution_budget(
        ActiveExecutionBudgetState(
            execution_id=execution_id,
            mode=ExecutionBudgetAllocationMode.SHARED,
            ledger=ledger,
        ),
    )
    return identity_token, authority_token, budget_token


def _metric_wiring_function_nodes() -> list[ast.AST]:
    tree = ast.parse(_NEXUS_LOOP_PATH.read_text(encoding="utf-8"))
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task":
            nodes.append(node)
    return nodes


def test_handle_task_metric_propagation_has_no_signature_inspection() -> None:
    nodes = _metric_wiring_function_nodes()
    assert nodes, "NexusLoop.handle_task must exist"
    handle_task = nodes[0]
    forbidden_names = {"signature", "getmembers"}
    for child in ast.walk(handle_task):
        if isinstance(child, ast.Attribute) and child.attr in forbidden_names:
            if isinstance(child.value, ast.Name) and child.value.id == "inspect":
                raise AssertionError(
                    "inspect-based architectural wiring is forbidden in handle_task",
                )
        if isinstance(child, ast.Name) and child.id == "impl_kwargs":
            raise AssertionError("dynamic kwargs probing is forbidden in handle_task")


def test_handle_task_calls_handle_task_impl_with_explicit_metric_scope_kwarg() -> None:
    tree = ast.parse(_NEXUS_LOOP_PATH.read_text(encoding="utf-8"))
    handle_task = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "handle_task"
    )
    found_explicit_call = False
    for child in ast.walk(handle_task):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not (
            isinstance(func, ast.Attribute)
            and func.attr == "_handle_task_impl"
            and isinstance(func.value, ast.Name)
            and func.value.id == "self"
        ):
            continue
        for keyword in child.keywords:
            if (
                keyword.arg == "runtime_event_metric_scope"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "metric_scope"
            ):
                found_explicit_call = True
    assert found_explicit_call


@pytest.mark.asyncio
async def test_handle_task_passes_invocation_metric_scope_to_impl() -> None:
    bus = RuntimeEventBus()
    loop = NexusLoop(AgentRegistry(), event_bus=bus)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()
    received: list[RuntimeEventMetricScope] = []

    async def _impl(
        task: Task,
        *,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        received.append(runtime_event_metric_scope)
        return TaskResult(
            task_id=task.task_id,
            run_id=run_id,
            state=TaskState.COMPLETED,
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    loop._handle_task_impl = _impl  # type: ignore[method-assign]
    id_t, auth_t, bud_t = _bind_nexus_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", message="m")
    try:
        await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_budget(bud_t)
        reset_active_execution_authority(auth_t)
        reset_active_execution_identity(id_t)
    assert len(received) == 1
    assert received[0] is not None
