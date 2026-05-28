# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.execution_phase import ExecutionPhase
from intergrax.contracts.memory_write_policy import MemoryWritePolicy
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.events.runtime_event import RuntimeEventType
from intergrax.runtime.task_memory import (
    InMemoryTaskMemoryStore,
    MemoryAccessPolicy,
    PolicyScopedMemoryView,
    TaskMemoryCoordinator,
)
from intergrax.runtime.task_memory.memory_view import MemoryViewAccessDenied


class _RecordingEmitter:
    def __init__(self) -> None:
        self.events = []

    async def emit(self, event) -> None:
        self.events.append(event)


def _view(
    *,
    store=None,
    emitter=None,
    policy=None,
    tenant_id="t1",
    task_id="task_1",
):
    store = store or InMemoryTaskMemoryStore()
    exec_ctx = RuntimeExecutionContext(
        task_id=task_id,
        run_id="run_1",
        agent_id="agent_a",
        phase=ExecutionPhase.STEP_EXECUTION,
        event_emitter=emitter,
    )
    return (
        PolicyScopedMemoryView(
            exec_ctx,
            store,
            tenant_id=tenant_id,
            task_id=task_id,
            access_policy=policy,
        ),
        store,
        exec_ctx,
    )


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_write_read_roundtrip():
    view, store, _ctx = _view(emitter=_RecordingEmitter())
    await view.write("findings", "vendor.a", {"score": 9})
    loaded = await view.read("findings", "vendor.a")
    assert loaded == {"score": 9}
    persisted = TaskMemoryCoordinator.read(
        store,
        tenant_id="t1",
        task_id="task_1",
        namespace="findings",
        key="vendor.a",
    )
    assert persisted is not None
    assert persisted.provenance["agent_id"] == "agent_a"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_merge_policy():
    view, _, _ctx = _view(emitter=_RecordingEmitter())
    await view.write("ns", "k", {"a": 1, "b": 1})
    await view.write("ns", "k", {"b": 2, "c": 3}, policy=MemoryWritePolicy.MERGE)
    loaded = await view.read("ns", "k")
    assert loaded == {"a": 1, "b": 2, "c": 3}


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_list_prefix():
    view, _, _ctx = _view(emitter=_RecordingEmitter())
    await view.write("findings", "vendor.a", {"score": 1})
    await view.write("findings", "vendor.b", {"score": 2})
    await view.write("findings", "summary", {"text": "ok"})
    rows = await view.list("findings", prefix="vendor.")
    assert [row.key for row in rows] == ["vendor.a", "vendor.b"]


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_emits_runtime_events():
    emitter = _RecordingEmitter()
    view, _, _ctx = _view(emitter=emitter)
    await view.write("ns", "k", {"v": 1})
    await view.read("ns", "k")
    types = [event.event_type for event in emitter.events]
    assert types.count(RuntimeEventType.MEMORY_WRITE) == 1
    assert types.count(RuntimeEventType.MEMORY_READ) == 1
    assert emitter.events[0].payload["namespace"] == "ns"
    assert emitter.events[0].tenant_id == "t1"


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_namespace_policy_denied():
    policy = MemoryAccessPolicy(allowed_namespaces=frozenset({"allowed"}))
    view, _, _ctx = _view(emitter=_RecordingEmitter(), policy=policy)
    with pytest.raises(MemoryViewAccessDenied, match="namespace not allowed"):
        await view.read("blocked", "k")


@pytest.mark.asyncio
@pytest.mark.unit
@pytest.mark.gate
async def test_memory_view_read_only_denies_write():
    policy = MemoryAccessPolicy(read_only=True)
    view, _, _ctx = _view(emitter=_RecordingEmitter(), policy=policy)
    with pytest.raises(MemoryViewAccessDenied, match="read-only"):
        await view.write("ns", "k", {"v": 1})


@pytest.mark.unit
@pytest.mark.gate
def test_memory_access_policy_from_metadata():
    from intergrax.runtime.task_memory.policy import memory_access_policy_from_metadata

    policy = memory_access_policy_from_metadata(
        {
            "memory_allowed_namespaces": ["vendor_report", "findings"],
            "memory_read_only": True,
        }
    )
    assert policy.read_only is True
    assert policy.allowed_namespaces == frozenset({"vendor_report", "findings"})
