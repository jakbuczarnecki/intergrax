# © Artur Czarnecki. All rights reserved.

"""HARDEN-1D — durable Problem Store failure & recovery through terminal diagnostics."""

from __future__ import annotations

import pytest

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_terminal_execution_diagnostic_trigger,
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.runtime.diagnostics.diagnostic_subsystem_failure_evidence import (
    diagnostic_subsystem_failure_observed_for_run,
    is_diagnostic_subsystem_failure_event,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.unified_task_runner import UnifiedTaskRunner
from testing_support.delegating_failing_conditional_document_store import (
    ControlledDocumentStoreWriteFailure,
    DelegatingFailingConditionalDocumentStore,
    DocumentStoreWriteFailureMode,
)

pytestmark = [pytest.mark.integration, pytest.mark.gate]

_TENANT = "harden-1d-tenant"


class _FakeEnvWiring:
    def __init__(self, document_store: object) -> None:
        self.build_context = _FakeBuildContext(document_store)


class _FakeBuildContext:
    def __init__(self, document_store: object) -> None:
        self.tool_wiring_context = _FakeToolWiringContext(document_store)


class _FakeToolWiringContext:
    def __init__(self, document_store: object) -> None:
        self.document_store = document_store


def _inject_violation_after_completed(
    runtime_store: InMemoryRuntimeEventStore,
    *,
    violating_event_type: RuntimeEventType,
):
    def _handler(event: RuntimeEvent) -> None:
        if event.event_type is not RuntimeEventType.TASK_COMPLETED:
            return
        runtime_store.append(
            sample_runtime_event(
                tenant_id=event.tenant_id,
                task_id=event.task_id,
                run_id=event.run_id,
                attempt_id=event.attempt_id,
            ).model_copy(update={"event_type": violating_event_type}),
            tenant_id=event.tenant_id,
        )

    return _handler


def _build_diagnostic_nexus_loop(
    *,
    document_store: DelegatingFailingConditionalDocumentStore,
    inject_violation: bool,
) -> tuple[NexusLoop, InMemoryRuntimeEventStore, object]:
    runtime_store = InMemoryRuntimeEventStore()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=_FakeEnvWiring(document_store),
        observability=stores,
    )
    assert deps is not None
    trigger = build_terminal_execution_diagnostic_trigger(deps)

    registry = AgentRegistry()
    registry.register(EchoAgent())
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=runtime_store,
    )
    loop.attach_terminal_diagnostic_trigger(trigger)
    if inject_violation:
        loop.event_bus.subscribe(
            _inject_violation_after_completed(
                runtime_store,
                violating_event_type=RuntimeEventType.RETRY_SCHEDULED,
            ),
            event_types={RuntimeEventType.TASK_COMPLETED},
            priority=10,
        )
    return loop, runtime_store, deps.problem_persistence


async def _run_echo(
    runner: UnifiedTaskRunner,
    *,
    message: str,
    run_id: object | None = None,
) -> object:
    return await runner.run_task(
        Task(
            tenant_id=_TENANT,
            user_id="user-1",
            message=message,
            context=TaskContext(capability="echo.basic"),
        ),
        run_id=run_id or mint_run_id(),
    )


@pytest.mark.asyncio
async def test_harden_1d_terminal_create_failure_preserves_business_result() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    loop, runtime_store, persistence = _build_diagnostic_nexus_loop(
        document_store=store,
        inject_violation=True,
    )
    runner = UnifiedTaskRunner(loop)
    run_id = mint_run_id()

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    result = await _run_echo(runner, message="harden-1d create failure", run_id=run_id)

    assert result.state is TaskState.COMPLETED
    assert query_all_problems_for_tenant(persistence, _TENANT) == ()
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=run_id,
    )
    failure_events = [
        event
        for event in runtime_store.list_for_run(run_id, tenant_id=_TENANT)
        if is_diagnostic_subsystem_failure_event(event)
    ]
    assert len(failure_events) == 1
    assert failure_events[0].payload["error_type"] == ControlledDocumentStoreWriteFailure.__name__


@pytest.mark.asyncio
async def test_harden_1d_terminal_update_failure_preserves_business_result() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    loop, runtime_store, persistence = _build_diagnostic_nexus_loop(
        document_store=store,
        inject_violation=True,
    )
    runner = UnifiedTaskRunner(loop)

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.HEALTHY)
    baseline = await _run_echo(runner, message="harden-1d baseline")
    assert baseline.state is TaskState.COMPLETED
    problems_after_baseline = query_all_problems_for_tenant(persistence, _TENANT)
    assert len(problems_after_baseline) == 1
    baseline_problem = problems_after_baseline[0]
    assert baseline_problem.occurrence_count == 1

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    failure_run_id = mint_run_id()
    failure_result = await _run_echo(
        runner,
        message="harden-1d update failure",
        run_id=failure_run_id,
    )
    assert failure_result.state is TaskState.COMPLETED
    unchanged = persistence.get(
        tenant_id=_TENANT,
        problem_id=baseline_problem.problem_id,
    )
    assert unchanged == baseline_problem
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=failure_run_id,
    )


@pytest.mark.asyncio
async def test_harden_1d_store_recovery_restores_diagnostic_persistence() -> None:
    store = DelegatingFailingConditionalDocumentStore()
    loop, runtime_store, persistence = _build_diagnostic_nexus_loop(
        document_store=store,
        inject_violation=True,
    )
    runner = UnifiedTaskRunner(loop)

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    failed_create_run_id = mint_run_id()
    failed_create = await _run_echo(
        runner,
        message="harden-1d failed create",
        run_id=failed_create_run_id,
    )
    assert failed_create.state is TaskState.COMPLETED
    assert query_all_problems_for_tenant(persistence, _TENANT) == ()
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=failed_create_run_id,
    )

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.HEALTHY)
    recovered_create = await _run_echo(runner, message="harden-1d recovered create")
    assert recovered_create.state is TaskState.COMPLETED
    problems_after_create = query_all_problems_for_tenant(persistence, _TENANT)
    assert len(problems_after_create) == 1
    assert problems_after_create[0].occurrence_count == 1

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.FAIL_WRITES)
    failed_update_run_id = mint_run_id()
    failed_update = await _run_echo(
        runner,
        message="harden-1d failed update",
        run_id=failed_update_run_id,
    )
    assert failed_update.state is TaskState.COMPLETED
    assert persistence.get(
        tenant_id=_TENANT,
        problem_id=problems_after_create[0].problem_id,
    ) == problems_after_create[0]
    assert diagnostic_subsystem_failure_observed_for_run(
        runtime_store,
        tenant_id=_TENANT,
        run_id=failed_update_run_id,
    )

    store.set_write_failure_mode(DocumentStoreWriteFailureMode.HEALTHY)
    recovered_update = await _run_echo(runner, message="harden-1d recovered update")
    assert recovered_update.state is TaskState.COMPLETED
    problems_after_update = query_all_problems_for_tenant(persistence, _TENANT)
    assert len(problems_after_update) == 1
    assert problems_after_update[0].problem_id == problems_after_create[0].problem_id
    assert problems_after_update[0].occurrence_count == 2
