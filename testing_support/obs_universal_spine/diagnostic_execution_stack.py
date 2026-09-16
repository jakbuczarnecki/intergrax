# © Artur Czarnecki. All rights reserved.

"""Reusable P3 stacks for OBS-UNIVERSAL-SPINE-E2E qualification (testing only)."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from echo.echo_agent import EchoAgent
from intergrax.applications._shared.diagnostic_read_wiring import HostDiagnosticReadDependencies
from intergrax.applications._shared.diagnostic_runtime_wiring import (
    build_terminal_execution_diagnostic_trigger,
    resolve_host_diagnostic_runtime_dependencies,
)
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.runtime.diagnostics.central_terminal_execution_diagnostic_port import (
    wrap_terminal_execution_diagnostic_trigger,
)
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.long_running.persistence_contract import TaskCheckpointPersistence
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.nexus.observability_wiring import wire_nexus_observability
from intergrax.runtime.observability.persistence_conformance import sample_runtime_event
from intergrax.runtime.registry.agent_registry import AgentRegistry

if TYPE_CHECKING:
    from intergrax.agents.agent_contract import Agent


class _FakeEnvWiring:
    def __init__(self, document_store: object) -> None:
        self.build_context = _FakeBuildContext(document_store)


class _FakeBuildContext:
    def __init__(self, document_store: object) -> None:
        self.tool_wiring_context = _FakeToolWiringContext(document_store)


class _FakeToolWiringContext:
    def __init__(self, document_store: object) -> None:
        self.document_store = document_store


def inject_violation_after_completed(
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


def build_diagnostic_nexus_loop(
    *,
    inject_violation: bool,
    violating_event_type: RuntimeEventType = RuntimeEventType.RETRY_SCHEDULED,
    problem_persistence: object | None = None,
    runtime_event_store: InMemoryRuntimeEventStore | None = None,
    checkpoint_store: TaskCheckpointPersistence | None = None,
    primary_agent: Agent | None = None,
) -> tuple[NexusLoop, InMemoryRuntimeEventStore, HostDiagnosticReadDependencies]:
    document_store = InMemoryDocumentStore()
    runtime_store = runtime_event_store or InMemoryRuntimeEventStore()
    stores = wire_nexus_observability(
        use_in_memory_trace=True,
        runtime_event_store=runtime_store,
    )
    deps = resolve_host_diagnostic_runtime_dependencies(
        env_wiring=_FakeEnvWiring(document_store),
        observability=stores,
    )
    assert deps is not None
    if problem_persistence is not None:
        deps = replace(deps, problem_persistence=problem_persistence)
    trigger = build_terminal_execution_diagnostic_trigger(deps)

    registry = AgentRegistry()
    registry.register(primary_agent or EchoAgent())
    loop = NexusLoop(
        registry,
        trace_store=stores.trace_store,
        runtime_event_store=runtime_store,
        checkpoint_store=checkpoint_store,
    )
    loop.attach_terminal_diagnostic_trigger(
        wrap_terminal_execution_diagnostic_trigger(trigger, event_bus=loop.event_bus),
    )
    if inject_violation:
        loop.event_bus.subscribe(
            inject_violation_after_completed(
                runtime_store,
                violating_event_type=violating_event_type,
            ),
            event_types={RuntimeEventType.TASK_COMPLETED},
            priority=10,
        )
    return loop, runtime_store, deps
