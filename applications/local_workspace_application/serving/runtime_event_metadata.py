# © Artur Czarnecki. All rights reserved.

"""Safe runtime TOOL_* event summary for LKW HTTP responses (LKW-PF1A)."""

from __future__ import annotations

import json
from typing import Any

from intergrax.runtime.events.persistence_contract import RuntimeEventPersistence
from intergrax.runtime.events.runtime_event import RuntimeEvent, RuntimeEventType
from intergrax.runtime.nexus.nexus_loop import NexusLoop
from intergrax.runtime.task.task import TaskResult

RUNTIME_EVENT_SUMMARY_KEY = "runtime_event_summary.v1"

_TOOL_EVENT_TYPES: frozenset[RuntimeEventType] = frozenset(
    {
        RuntimeEventType.TOOL_REQUESTED,
        RuntimeEventType.TOOL_COMPLETED,
        RuntimeEventType.TOOL_DENIED,
        RuntimeEventType.TOOL_FAILED,
    }
)

_BY_TYPE_KEYS: tuple[str, ...] = (
    "TOOL_REQUESTED",
    "TOOL_COMPLETED",
    "TOOL_DENIED",
    "TOOL_FAILED",
)

_UNSAFE_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "input",
        "query",
        "query_text",
        "text",
        "content",
        "raw_chunks",
        "chunks",
        "document",
        "documents",
        "source_path",
        "body",
        "message",
        "prompt",
        "redacted_input_summary",
    }
)


def build_runtime_event_summary(events: list[RuntimeEvent]) -> dict[str, Any]:
    """Aggregate TOOL_* runtime events into a redacted HTTP-safe summary."""
    tool_events = [event for event in events if event.event_type in _TOOL_EVENT_TYPES]
    by_type = {key: 0 for key in _BY_TYPE_KEYS}
    tools: dict[str, dict[str, Any]] = {}

    for event in tool_events:
        type_key = event.event_type.name
        if type_key in by_type:
            by_type[type_key] += 1

        payload = event.payload if isinstance(event.payload, dict) else {}
        tool_id = _tool_id_from_payload(payload)
        entry = tools.setdefault(
            tool_id,
            {
                "tool_id": tool_id,
                "requested": 0,
                "completed": 0,
                "denied": 0,
                "failed": 0,
            },
        )
        if event.event_type is RuntimeEventType.TOOL_REQUESTED:
            entry["requested"] += 1
        elif event.event_type is RuntimeEventType.TOOL_COMPLETED:
            entry["completed"] += 1
        elif event.event_type is RuntimeEventType.TOOL_DENIED:
            entry["denied"] += 1
        elif event.event_type is RuntimeEventType.TOOL_FAILED:
            entry["failed"] += 1

    return {
        "schema_version": RUNTIME_EVENT_SUMMARY_KEY,
        "tool_events": {
            "total": len(tool_events),
            "by_type": by_type,
            "tools": sorted(tools.values(), key=lambda item: str(item["tool_id"])),
        },
    }


def runtime_event_summary_is_safe(payload: dict[str, Any]) -> bool:
    """Return False when summary payload exposes raw content fields."""
    serialized = json.dumps(payload)
    for key in _UNSAFE_PAYLOAD_KEYS:
        if f'"{key}"' in serialized:
            return False
    tool_events = payload.get("tool_events")
    if not isinstance(tool_events, dict):
        return False
    for tool_entry in tool_events.get("tools") or []:
        if not isinstance(tool_entry, dict):
            continue
        if _UNSAFE_PAYLOAD_KEYS.intersection(tool_entry.keys()):
            return False
    return True


def collect_runtime_events_for_task(
    *,
    nexus_loop: NexusLoop | None,
    task_result: TaskResult,
    tenant_id: str,
) -> list[RuntimeEvent]:
    """Resolve runtime events for a completed task from bus history or persistence."""
    if nexus_loop is None:
        return []

    run_id = task_result.run_id
    task_id = task_result.task_id
    merged: dict[str, RuntimeEvent] = {}

    for event in nexus_loop.event_bus.history:
        if _event_matches_task(event, run_id=run_id, task_id=task_id):
            merged[event.event_id] = event

    store: RuntimeEventPersistence | None = nexus_loop.runtime_event_store
    if store is not None:
        for source in (
            store.list_for_run(run_id, tenant_id=tenant_id),
            store.list_for_task(task_id, tenant_id=tenant_id),
        ):
            for event in source:
                merged.setdefault(event.event_id, event)

    return list(merged.values())


def attach_runtime_event_summary_metadata(
    metadata: dict[str, Any],
    *,
    task_result: TaskResult,
    nexus_loop: NexusLoop | None,
    tenant_id: str,
) -> dict[str, Any]:
    """Attach curated ``runtime_event_summary.v1`` derived from platform runtime events."""
    events = collect_runtime_events_for_task(
        nexus_loop=nexus_loop,
        task_result=task_result,
        tenant_id=tenant_id,
    )
    summary = build_runtime_event_summary(events)
    metadata[RUNTIME_EVENT_SUMMARY_KEY] = summary
    return metadata


def _event_matches_task(
    event: RuntimeEvent,
    *,
    run_id: str,
    task_id: str,
) -> bool:
    ids = {run_id, task_id}
    if event.run_id:
        if event.run_id in ids:
            return True
        if any(event.run_id == candidate or event.run_id.startswith(f"{candidate}:") for candidate in ids):
            return True
    if event.task_id and event.task_id in ids:
        return True
    return False


def _tool_id_from_payload(payload: dict[str, Any]) -> str:
    for key in ("tool_id", "tool_name"):
        raw = payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return "unknown"
