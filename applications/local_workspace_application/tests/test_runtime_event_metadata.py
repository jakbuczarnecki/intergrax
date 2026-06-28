# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import TaskResult, TaskState
from intergrax.tools.providers.rag.ingest_service import RAG_INGEST_TOOL_ID
from intergrax.tools.providers.rag.service import RAG_TOOL_ID
from intergrax.tools.providers.workspace.service import WORKSPACE_WRITE_FILE_TOOL_ID
from local_workspace_application.serving.runtime_event_metadata import (
    RUNTIME_EVENT_SUMMARY_KEY,
    attach_runtime_event_summary_metadata,
    build_runtime_event_summary,
    collect_runtime_events_for_task,
    runtime_event_summary_is_safe,
)

_RAW_INPUT = {
    "source_path": "/tmp/secret-doc.txt",
    "query": "project X",
    "text": "raw chunk body",
}


def _tool_event(
    event_type: RuntimeEventType,
    *,
    tool_id: str,
    run_id: str = "run-1",
    task_id: str = "task-1",
) -> RuntimeEvent:
    return RuntimeEvent(
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": tool_id,
            "status": event_type.name.removeprefix("TOOL_").lower(),
            "args_digest": "abc123",
            **_RAW_INPUT,
        },
        run_id=run_id,
        task_id=task_id,
    )


@pytest.mark.unit
def test_build_runtime_event_summary_aggregates_tool_counts() -> None:
    events = [
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_INGEST_TOOL_ID),
        _tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_INGEST_TOOL_ID),
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID),
        _tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_TOOL_ID),
    ]

    summary = build_runtime_event_summary(events)

    assert summary["schema_version"] == RUNTIME_EVENT_SUMMARY_KEY
    tool_events = summary["tool_events"]
    assert tool_events["total"] == 4
    assert tool_events["by_type"]["TOOL_REQUESTED"] == 2
    assert tool_events["by_type"]["TOOL_COMPLETED"] == 2
    tools = {entry["tool_id"]: entry for entry in tool_events["tools"]}
    assert tools[RAG_INGEST_TOOL_ID]["requested"] == 1
    assert tools[RAG_INGEST_TOOL_ID]["completed"] == 1
    assert tools[RAG_TOOL_ID]["requested"] == 1
    assert tools[RAG_TOOL_ID]["completed"] == 1


@pytest.mark.unit
def test_runtime_event_summary_excludes_raw_payload_fields() -> None:
    events = [
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=WORKSPACE_WRITE_FILE_TOOL_ID),
        _tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=WORKSPACE_WRITE_FILE_TOOL_ID),
    ]

    summary = build_runtime_event_summary(events)

    assert runtime_event_summary_is_safe(summary)
    serialized = json.dumps(summary)
    for raw_value in _RAW_INPUT.values():
        assert raw_value not in serialized


@pytest.mark.unit
def test_collect_runtime_events_for_task_filters_event_bus_history() -> None:
    bus = RuntimeEventBus(record_history=True)
    bus.record(_tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID, run_id="run-a"))
    bus.record(_tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_TOOL_ID, run_id="run-a"))
    bus.record(_tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_INGEST_TOOL_ID, run_id="run-b"))

    nexus_loop = NexusLoop.__new__(NexusLoop)
    object.__setattr__(nexus_loop, "_event_bus", bus)
    object.__setattr__(nexus_loop, "_runtime_event_store", None)

    task_result = TaskResult(
        task_id="task-a",
        run_id="run-a",
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )

    events = collect_runtime_events_for_task(
        nexus_loop=nexus_loop,
        task_result=task_result,
        tenant_id="default",
    )

    assert len(events) == 2
    assert {event.event_type for event in events} == {
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
    }


@pytest.mark.unit
def test_attach_runtime_event_summary_metadata() -> None:
    bus = RuntimeEventBus(record_history=True)
    bus.record(_tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID))
    bus.record(_tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_TOOL_ID))

    nexus_loop = NexusLoop.__new__(NexusLoop)
    object.__setattr__(nexus_loop, "_event_bus", bus)
    object.__setattr__(nexus_loop, "_runtime_event_store", None)

    task_result = TaskResult(
        task_id="task-1",
        run_id="run-1",
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )
    metadata: dict[str, object] = {"lkw_evidence.v1": {"schema_version": "lkw_evidence.v1"}}

    attach_runtime_event_summary_metadata(
        metadata,
        task_result=task_result,
        nexus_loop=nexus_loop,
        tenant_id="default",
    )

    summary = metadata[RUNTIME_EVENT_SUMMARY_KEY]
    assert isinstance(summary, dict)
    assert summary["tool_events"]["total"] == 2
    assert metadata["lkw_evidence.v1"]["schema_version"] == "lkw_evidence.v1"
