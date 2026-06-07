# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.task_memory.models import TaskMemoryRecord
from intergrax.tools.registry.runtime_bindings import TaskMemoryViewBinding
from intergrax.tools._shared.async_dispatch import run_async
from intergrax.tools.providers.memory.contracts import (
    MemoryKeyRecord,
    MemoryListKeysInput,
    MemoryListKeysOutput,
    MemoryReadInput,
    MemoryReadOutput,
    MemoryWriteInput,
    MemoryWriteOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

MEMORY_READ_TOOL_ID = "memory.read"
MEMORY_WRITE_TOOL_ID = "memory.write"
MEMORY_LIST_KEYS_TOOL_ID = "memory.list_keys"


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
