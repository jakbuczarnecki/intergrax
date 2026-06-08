# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.tools.registry.runtime_bindings import TaskMemoryViewBinding
from intergrax.tools._shared.async_dispatch import run_async
from intergrax.tools.providers.memory.contracts import (
    MemoryDeleteKeyInput,
    MemoryDeleteKeyOutput,
    MemoryKeyRecord,
    MemoryListKeysInput,
    MemoryListKeysOutput,
    MemoryReadInput,
    MemoryReadOutput,
    MemorySearchInput,
    MemorySearchMatch,
    MemorySearchOutput,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

MEMORY_READ_TOOL_ID = "memory.read"
MEMORY_WRITE_TOOL_ID = "memory.write"
MEMORY_LIST_KEYS_TOOL_ID = "memory.list_keys"
MEMORY_DELETE_KEY_TOOL_ID = "memory.delete_key"
MEMORY_SEARCH_TOOL_ID = "memory.search"


def _require_memory_view(ctx: ToolWiringContext) -> TaskMemoryViewBinding:
    view = ctx.memory_view
    if view is None:
        raise RuntimeError("memory_view_not_configured")
    return view


def _to_key_record(record: TaskMemoryRecord) -> MemoryKeyRecord:
    return MemoryKeyRecord(
        key=record.key,
        record_id=record.record_id,
        updated_at_utc=record.updated_at_utc,
    )


def memory_read(ctx: ToolWiringContext, params: MemoryReadInput) -> MemoryReadOutput:
    view = _require_memory_view(ctx)
    value = run_async(view.read(params.namespace.strip(), params.key.strip()))
    found = value is not None
    return MemoryReadOutput(
        namespace=params.namespace.strip(),
        key=params.key.strip(),
        found=found,
        value=dict(value or {}),
    )


def memory_write(ctx: ToolWiringContext, params: MemoryWriteInput) -> MemoryWriteOutput:
    view = _require_memory_view(ctx)
    run_async(
        view.write(
            params.namespace.strip(),
            params.key.strip(),
            dict(params.value),
            policy=params.policy,
        )
    )
    return MemoryWriteOutput(
        namespace=params.namespace.strip(),
        key=params.key.strip(),
        written=True,
    )


def memory_list_keys(ctx: ToolWiringContext, params: MemoryListKeysInput) -> MemoryListKeysOutput:
    view = _require_memory_view(ctx)
    records = run_async(view.list(params.namespace.strip(), params.prefix))
    keys: list[MemoryKeyRecord] = []
    for record in records:
        if isinstance(record, TaskMemoryRecord):
            keys.append(_to_key_record(record))
        elif isinstance(record, dict):
            keys.append(
                MemoryKeyRecord(
                    key=str(record.get("key", "")),
                    record_id=str(record.get("record_id", "")),
                    updated_at_utc=str(record.get("updated_at_utc", "")),
                )
            )
    return MemoryListKeysOutput(
        namespace=params.namespace.strip(),
        prefix=params.prefix,
        keys=keys,
        total=len(keys),
    )


def _value_matches_query(value: dict, query: str) -> bool:
    needle = query.lower()
    for key, item in value.items():
        if needle in str(key).lower():
            return True
        if needle in str(item).lower():
            return True
    return False


def memory_search(ctx: ToolWiringContext, params: MemorySearchInput) -> MemorySearchOutput:
    view = _require_memory_view(ctx)
    records = run_async(view.list(params.namespace.strip(), params.prefix))
    query = params.query.strip()
    matches: list[MemorySearchMatch] = []
    for record in records:
        key = ""
        if isinstance(record, TaskMemoryRecord):
            key = record.key
        elif isinstance(record, dict):
            key = str(record.get("key") or "")
        else:
            continue
        if query.lower() not in key.lower():
            value = run_async(view.read(params.namespace.strip(), key)) or {}
            if not _value_matches_query(dict(value), query):
                continue
        else:
            value = run_async(view.read(params.namespace.strip(), key)) or {}
        matches.append(MemorySearchMatch(key=key, value=dict(value)))
        if len(matches) >= params.limit:
            break
    return MemorySearchOutput(
        namespace=params.namespace.strip(),
        query=query,
        matches=matches,
        total=len(matches),
    )


def memory_delete_key(ctx: ToolWiringContext, params: MemoryDeleteKeyInput) -> MemoryDeleteKeyOutput:
    view = _require_memory_view(ctx)
    deleted = run_async(view.delete(params.namespace.strip(), params.key.strip()))
    return MemoryDeleteKeyOutput(
        namespace=params.namespace.strip(),
        key=params.key.strip(),
        deleted=bool(deleted),
    )
