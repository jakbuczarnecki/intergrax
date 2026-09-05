# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_event_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.events.stores.memory_runtime_event_store import InMemoryRuntimeEventStore
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
    run_id: str | None = None,
    task_id: str | None = None,
    tenant_id: str = "default",
    event_id: str | None = None,
) -> RuntimeEvent:
    resolved_run_id = run_id or mint_run_id()
    resolved_task_id = task_id or mint_task_id()
    return RuntimeEvent(
        event_id=event_id or mint_event_id(),
        event_type=event_type,
        phase=ExecutionPhase.STEP_EXECUTION,
        payload={
            "tool_id": tool_id,
            "status": event_type.name.removeprefix("TOOL_").lower(),
            "args_digest": "abc123",
            **_RAW_INPUT,
        },
        run_id=resolved_run_id,
        task_id=resolved_task_id,
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id(),
        tenant_id=tenant_id,
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
def test_collect_runtime_events_for_task_reads_persistence_for_task() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    store.append(
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="default",
    )
    store.append(
        _tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="default",
    )
    store.append(
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_INGEST_TOOL_ID, run_id=mint_run_id()),
        tenant_id="default",
    )

    task_result = TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )

    events = collect_runtime_events_for_task(
        runtime_event_persistence=store,
        task_result=task_result,
        tenant_id="default",
    )

    assert len(events) == 2
    assert {event.event_type for event in events} == {
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
    }


@pytest.mark.unit
def test_collect_runtime_events_for_task_scopes_by_tenant() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    store.append(
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="tenant-a",
    )
    store.append(
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="tenant-b",
    )

    task_result = TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )

    events = collect_runtime_events_for_task(
        runtime_event_persistence=store,
        task_result=task_result,
        tenant_id="tenant-a",
    )

    assert len(events) == 1


@pytest.mark.unit
def test_collect_runtime_events_for_task_deduplicates_by_event_id() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    event_id = mint_event_id()
    event = _tool_event(
        RuntimeEventType.TOOL_REQUESTED,
        tool_id=RAG_TOOL_ID,
        run_id=run_id,
        task_id=task_id,
        event_id=event_id,
    )
    store.append(event, tenant_id="default")
    store.append(event, tenant_id="default")

    task_result = TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )

    events = collect_runtime_events_for_task(
        runtime_event_persistence=store,
        task_result=task_result,
        tenant_id="default",
    )

    assert len(events) == 1


@pytest.mark.unit
def test_collect_runtime_events_for_task_includes_child_agent_run_ids() -> None:
    store = InMemoryRuntimeEventStore()
    child_run_id = mint_run_id()
    store.append(
        _tool_event(
            RuntimeEventType.TOOL_DENIED,
            tool_id=RAG_TOOL_ID,
            run_id=child_run_id,
        ),
        tenant_id="default",
    )

    task_result = TaskResult(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
        metadata={
            "application_run_summary.v1": {
                "agent_invocations": [{"run_id": child_run_id}],
            }
        },
    )

    events = collect_runtime_events_for_task(
        runtime_event_persistence=store,
        task_result=task_result,
        tenant_id="default",
    )

    assert len(events) == 1
    assert events[0].event_type is RuntimeEventType.TOOL_DENIED


@pytest.mark.unit
def test_attach_runtime_event_summary_metadata() -> None:
    store = InMemoryRuntimeEventStore()
    run_id = mint_run_id()
    task_id = mint_task_id()
    store.append(
        _tool_event(RuntimeEventType.TOOL_REQUESTED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="default",
    )
    store.append(
        _tool_event(RuntimeEventType.TOOL_COMPLETED, tool_id=RAG_TOOL_ID, run_id=run_id, task_id=task_id),
        tenant_id="default",
    )

    task_result = TaskResult(
        task_id=task_id,
        run_id=run_id,
        state=TaskState.COMPLETED,
        answer="done",
        agent_id="local_search",
    )
    metadata: dict[str, object] = {"lkw_evidence.v1": {"schema_version": "lkw_evidence.v1"}}

    attach_runtime_event_summary_metadata(
        metadata,
        task_result=task_result,
        runtime_event_persistence=store,
        tenant_id="default",
    )

    summary = metadata[RUNTIME_EVENT_SUMMARY_KEY]
    assert isinstance(summary, dict)
    assert summary["tool_events"]["total"] == 2
    assert metadata["lkw_evidence.v1"]["schema_version"] == "lkw_evidence.v1"
