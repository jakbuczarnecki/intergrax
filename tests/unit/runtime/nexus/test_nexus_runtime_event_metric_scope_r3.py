# © Artur Czarnecki. All rights reserved.

"""OBS-RUNTIME-HISTORY-BOUNDS-R3: invocation-local Nexus metric scopes."""

from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_identity import (
    bind_active_execution_identity,
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
    reset_active_execution_identity,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.runtime_event_metric_scope import (
    RuntimeEventMetricScope,
    _RuntimeEventMetricScopeRegistry,
)
from intergrax.runtime.governance.active_execution_authority import (
    bind_active_execution_authority,
    reset_active_execution_authority,
)
from intergrax.runtime.execution.active_execution_budget import (
    ActiveExecutionBudgetState,
    bind_active_execution_budget,
    reset_active_execution_budget,
)
from intergrax.runtime.execution.budget.ledger import create_execution_budget_ledger
from intergrax.runtime.execution.budget.models import ExecutionBudgetAllocationMode
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskResult, TaskState
from intergrax.runtime.task.task_result_authoritative_exposure_defaults import (
    terminal_task_result_exposure_no_decision_gate,
)
from intergrax.contracts.delegation_authority import ParentExecutionAuthority
from intergrax.contracts.execution_identity import mint_event_id
from intergrax.contracts.runtime_event_history import RuntimeEventHistoryPolicy

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _event(*, label: str, task_id: str, run_id: str) -> RuntimeEvent:
    return RuntimeEvent.model_validate(
        {
            "tenant_id": "tenant-a",
            "task_id": task_id,
            "run_id": run_id,
            "attempt_id": mint_attempt_id(),
            "execution_id": mint_execution_id(),
            "event_id": mint_event_id(),
            "event_type": RuntimeEventType.STEP_STARTED,
            "phase": ExecutionPhase.STEP_EXECUTION,
            "payload": {"label": label},
        },
    )


def _bind_nexus_context(*, run_id: str, attempt_id: str, execution_id: str):
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


@pytest.mark.asyncio
async def test_deterministic_concurrent_handle_task_metric_isolation() -> None:
    bus = RuntimeEventBus(history_policy=RuntimeEventHistoryPolicy.bounded(2))
    loop = NexusLoop(AgentRegistry(), event_bus=bus)
    run_a = mint_run_id()
    run_b = mint_run_id()
    task_a = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="a1",
        message="A",
        task_id=mint_task_id(),
    )
    task_b = Task(
        tenant_id="t1",
        user_id="u1",
        agent_id="a1",
        message="B",
        task_id=mint_task_id(),
    )
    a_emitted = asyncio.Event()
    b_emitted = asyncio.Event()
    a_may_finish = asyncio.Event()

    async def _impl(
        task: Task,
        *,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        if task.message == "A":
            bus.record(_event(label="a1", task_id=task.task_id, run_id=run_a))
            a_emitted.set()
            await b_emitted.wait()
            await a_may_finish.wait()
            return TaskResult(
                task_id=task.task_id,
                run_id=run_a,
                state=TaskState.COMPLETED,
                authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
            )
        await a_emitted.wait()
        for index in range(3):
            bus.record(
                _event(label=f"b{index}", task_id=task.task_id, run_id=run_b),
            )
        b_emitted.set()
        await asyncio.sleep(0)
        return TaskResult(
            task_id=task.task_id,
            run_id=run_b,
            state=TaskState.COMPLETED,
            authoritative_decision_exposure=terminal_task_result_exposure_no_decision_gate(),
        )

    loop._handle_task_impl = _impl  # type: ignore[method-assign]

    exec_a = mint_execution_id()
    exec_b = mint_execution_id()
    attempt_a = mint_attempt_id()
    attempt_b = mint_attempt_id()

    async def _run_a() -> TaskResult:
        id_t, auth_t, bud_t = _bind_nexus_context(
            run_id=run_a,
            attempt_id=attempt_a,
            execution_id=exec_a,
        )
        try:
            return await loop.handle_task(task_a, run_id=run_a, attempt_id=attempt_a)
        finally:
            reset_active_execution_budget(bud_t)
            reset_active_execution_authority(auth_t)
            reset_active_execution_identity(id_t)

    async def _run_b() -> TaskResult:
        id_t, auth_t, bud_t = _bind_nexus_context(
            run_id=run_b,
            attempt_id=attempt_b,
            execution_id=exec_b,
        )
        try:
            return await loop.handle_task(task_b, run_id=run_b, attempt_id=attempt_b)
        finally:
            reset_active_execution_budget(bud_t)
            reset_active_execution_authority(auth_t)
            reset_active_execution_identity(id_t)

    task_a_handle = asyncio.create_task(_run_a())
    await a_emitted.wait()
    task_b_handle = asyncio.create_task(_run_b())
    await b_emitted.wait()
    a_may_finish.set()
    result_a = await task_a_handle
    result_b = await task_b_handle
    assert result_a.summary.metrics.runtime_events == 1
    assert result_b.summary.metrics.runtime_events == 3


def test_metric_scope_close_is_idempotent() -> None:
    bus = RuntimeEventBus()
    task_id = mint_task_id()
    run_id = mint_run_id()
    scope = bus.open_runtime_event_metric_scope(task_id, run_id)
    scope.close()
    scope.close()


def test_metric_registry_no_leak_after_many_scopes() -> None:
    bus = RuntimeEventBus()
    task_id = mint_task_id()
    for _ in range(1000):
        run_id = mint_run_id()
        scope = bus.open_runtime_event_metric_scope(task_id, run_id)
        bus.record(_event(label="x", task_id=task_id, run_id=run_id))
        scope.close()
    registry: _RuntimeEventMetricScopeRegistry = bus._metric_scopes  # noqa: SLF001
    assert registry._scopes == {}  # noqa: SLF001


@pytest.mark.asyncio
async def test_metric_scope_cleaned_up_when_handle_task_raises() -> None:
    bus = RuntimeEventBus()
    loop = NexusLoop(AgentRegistry(), event_bus=bus)
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    async def _boom(
        task: Task,
        *,
        runtime_event_metric_scope: RuntimeEventMetricScope,
    ) -> TaskResult:
        bus.record(_event(label="x", task_id=task.task_id, run_id=run_id))
        raise RuntimeError("boom")

    loop._handle_task_impl = _boom  # type: ignore[method-assign]
    id_t, auth_t, bud_t = _bind_nexus_context(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", message="m")
    try:
        with pytest.raises(RuntimeError, match="boom"):
            await loop.handle_task(task, run_id=run_id, attempt_id=attempt_id)
    finally:
        reset_active_execution_budget(bud_t)
        reset_active_execution_authority(auth_t)
        reset_active_execution_identity(id_t)
    registry: _RuntimeEventMetricScopeRegistry = bus._metric_scopes  # noqa: SLF001
    assert registry._scopes == {}  # noqa: SLF001
