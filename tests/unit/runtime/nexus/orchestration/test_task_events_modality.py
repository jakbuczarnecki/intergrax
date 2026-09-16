from __future__ import annotations

import asyncio

import pytest

from intergrax.contracts.execution_identity import ActiveExecutionIdentity
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
from intergrax.runtime.events.unified_run_journal import JOURNAL_SCHEMA_VERSION
from intergrax.runtime.nexus.orchestration.task_events import NexusRuntimeEventPublisher
from intergrax.runtime.nexus.tracing.in_memory_trace_store import InMemoryRunTraceStore
from intergrax.runtime.nexus.tracing.persistence_models import RunMetadata, RunStats, SerializedTraceEvent
from intergrax.runtime.observability.modality_metrics import extract_modality_metrics
from intergrax.runtime.task.task import TaskState
from testing_support.builder import (
    build_task_for_tests,
    canonical_governed_execution_scope,
    canonical_run_id_for_tests,
    canonical_task_id_for_tests,
)


def _finalize_run(
    store: InMemoryRunTraceStore,
    *,
    store_key: str,
    run_id: str,
    tenant_id: str,
) -> None:
    store.finalize_run(
        store_key,
        RunMetadata(
            run_id=run_id,
            tenant_id=tenant_id,
            user_id="user-1",
            session_id="session-1",
            started_at_utc="2026-01-01T00:00:00Z",
            stats=RunStats(duration_ms=10, llm_usage={}),
        ),
    )


@pytest.mark.asyncio
async def test_publish_terminal_attaches_modality_metrics_from_trace() -> None:
    store = InMemoryRunTraceStore()
    seed = "run-modality-1"
    run_id = str(canonical_run_id_for_tests(seed))
    task_id = str(canonical_task_id_for_tests(seed))
    tenant_id = "tenant-a"
    store._events_by_run[task_id] = [
        SerializedTraceEvent(
            event_id="e1",
            run_id=run_id,
            seq=1,
            ts_utc="2026-01-01T00:00:00Z",
            level="info",
            component="tools",
            step="tool_invocation_end",
            message="done",
            payload_schema_id=None,
            payload_schema_version=None,
            payload={"modality_metrics": {"inference_ms": 15, "vision_detections": 1}},
            tags={},
            artifact_refs=[],
        ),
    ]
    _finalize_run(store, store_key=task_id, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus()
    received: list = []

    async def _capture(event) -> None:
        received.append(event)

    bus.subscribe(_capture, event_types={RuntimeEventType.TASK_COMPLETED})

    task = build_task_for_tests(
        seed=seed,
        tenant_id=tenant_id,
        user_id="user-1",
    ).model_copy(update={"state": TaskState.COMPLETED})
    publisher = NexusRuntimeEventPublisher(
        bus,
        current_task=lambda: task,
        trace_reader=store,
        execution_identity=ActiveExecutionIdentity(),
    )
    with canonical_governed_execution_scope(seed):
        await publisher.publish_terminal(task)

    assert len(received) == 1
    metrics = extract_modality_metrics(received[0])
    assert metrics.inference_ms == 15
    assert metrics.vision_detections == 1


@pytest.mark.asyncio
async def test_publish_terminal_without_trace_reader_omits_modality_metrics() -> None:
    bus = RuntimeEventBus()
    received: list = []

    async def _capture(event) -> None:
        received.append(event)

    bus.subscribe(_capture, event_types={RuntimeEventType.TASK_COMPLETED})

    modality_seed = "run-modality-2"
    task = build_task_for_tests(
        seed=modality_seed,
        tenant_id="tenant-a",
        user_id="user-1",
    ).model_copy(update={"state": TaskState.COMPLETED})
    publisher = NexusRuntimeEventPublisher(
        bus,
        current_task=lambda: task,
        execution_identity=ActiveExecutionIdentity(),
    )
    with canonical_governed_execution_scope(modality_seed):
        await publisher.publish_terminal(task)

    assert len(received) == 1
    assert "modality_metrics" not in received[0].payload


@pytest.mark.asyncio
async def test_publish_terminal_attaches_journal_ref_from_unified_journal() -> None:
    store = InMemoryRunTraceStore()
    seed = "run-journal-ref-1"
    run_id = str(canonical_run_id_for_tests(seed))
    tenant_id = "tenant-a"
    task_id = str(canonical_task_id_for_tests(seed))
    store._events_by_run[task_id] = [
        SerializedTraceEvent(
            event_id="e-life",
            run_id=run_id,
            seq=1,
            ts_utc="2026-01-01T00:00:00Z",
            level="info",
            component="planner",
            step="task_lifecycle",
            message="task state -> completed",
            payload_schema_id=None,
            payload_schema_version=None,
            payload=None,
            tags={"task_id": str(canonical_task_id_for_tests(seed)), "task_state": "completed"},
            artifact_refs=[],
        ),
    ]
    _finalize_run(store, store_key=task_id, run_id=run_id, tenant_id=tenant_id)

    bus = RuntimeEventBus()
    received: list = []

    async def _capture(event) -> None:
        received.append(event)

    bus.subscribe(_capture, event_types={RuntimeEventType.TASK_COMPLETED})

    task = build_task_for_tests(
        seed=seed,
        tenant_id=tenant_id,
        user_id="user-1",
    ).model_copy(update={"state": TaskState.COMPLETED})
    runtime_store = InMemoryRuntimeEventStore()
    publisher = NexusRuntimeEventPublisher(
        bus,
        current_task=lambda: task,
        trace_reader=store,
        runtime_event_store=runtime_store,
        execution_identity=ActiveExecutionIdentity(),
    )
    with canonical_governed_execution_scope(seed):
        await publisher.publish_terminal(task)

    assert len(received) == 1
    journal_ref = received[0].payload.get("journal_ref")
    assert isinstance(journal_ref, dict)
    assert journal_ref["schema_version"] == JOURNAL_SCHEMA_VERSION
    assert journal_ref["run_id"] == run_id
    assert journal_ref["event_count"] == 0
    assert journal_ref["parser_trace_count"] == 0
