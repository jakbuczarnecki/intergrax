# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from intergrax.runtime.task_memory import InMemoryTaskMemoryStore, MemoryAccessPolicy, PolicyScopedMemoryView
from intergrax.runtime.task_memory.coordinator import TaskMemoryCoordinator
from intergrax.runtime.task_memory.memory_view import MemoryViewAccessDenied
from intergrax.runtime.task_memory.models import TaskMemoryRecord
from testing_support.builder import build_runtime_execution_context_for_tests

pytestmark = pytest.mark.gate

_TENANT_ID = "t1"


def _view_with_policy(**kwargs):
    store = kwargs.pop("store", None) or InMemoryTaskMemoryStore()
    policy = kwargs.pop("policy", None)
    retention_days = kwargs.pop("retention_days", None)
    tenant_id = kwargs.pop("tenant_id", _TENANT_ID)
    metadata = kwargs.pop("metadata", None)
    exec_ctx = build_runtime_execution_context_for_tests(
        metadata=metadata or {},
        tenant_id=tenant_id,
    )
    task_id = exec_ctx.task_id
    view = PolicyScopedMemoryView(
        exec_ctx,
        store,
        access_policy=policy or MemoryAccessPolicy(),
        retention_days=retention_days,
    )
    return view, store, exec_ctx


@pytest.mark.asyncio
async def test_list_hides_stm_records_under_retention() -> None:
    store = InMemoryTaskMemoryStore()
    stale = datetime.now(timezone.utc) - timedelta(days=40)
    view, store, exec_ctx = _view_with_policy(retention_days=30)
    task_id = exec_ctx.task_id
    store.put(
        TaskMemoryRecord(
            tenant_id=_TENANT_ID,
            task_id=task_id,
            namespace="stm:scratch",
            key="old",
            value={"v": 1},
            updated_at_utc=stale.isoformat(),
            created_at_utc=stale.isoformat(),
        )
    )
    TaskMemoryCoordinator.write(
        store,
        tenant_id=_TENANT_ID,
        task_id=task_id,
        namespace="stm:scratch",
        key="fresh",
        value={"v": 2},
    )

    rows = await view.list("stm:scratch")
    assert [row.key for row in rows] == ["fresh"]


@pytest.mark.asyncio
async def test_scope_boundary_applies_to_read_list_delete() -> None:
    policy = MemoryAccessPolicy(scope_boundary="tenant")
    view, _, exec_ctx = _view_with_policy(policy=policy)
    await view.write("ns", "k", {"v": 1})
    exec_ctx.metadata["tenant_id"] = "other-tenant"
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.read("ns", "k")
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.list("ns")
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.delete("ns", "k")
    with pytest.raises(MemoryViewAccessDenied, match="conflicts"):
        await view.write("ns", "k2", {"v": 2})
