# © Artur Czarnecki. All rights reserved.

"""MEM-7.2: memory write hooks can block or mutate writes."""

from __future__ import annotations

import pytest

from intergrax.runtime.hooks.hook_context import HookAction, HookContext, HookResult
from intergrax.runtime.hooks.hook_point import HookPoint
from intergrax.runtime.hooks.hook_registry import HookRegistry
from intergrax.runtime.task_memory import InMemoryTaskMemoryStore, PolicyScopedMemoryView
from intergrax.runtime.task_memory.memory_view import MemoryViewAccessDenied
from testing_support.builder import build_runtime_execution_context_for_tests

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _view(*, hook_registry: HookRegistry | None = None) -> PolicyScopedMemoryView:
    exec_ctx = build_runtime_execution_context_for_tests(
        seed="hook",
        agent_id="agent_hook",
        tenant_id="tenant-hook",
    )
    return PolicyScopedMemoryView(
        exec_ctx,
        InMemoryTaskMemoryStore(),
        hook_registry=hook_registry,
    )


@pytest.mark.asyncio
async def test_before_memory_write_hook_blocks_persist() -> None:
    registry = HookRegistry()

    def _block(_ctx: HookContext) -> HookResult:
        return HookResult(action=HookAction.BLOCK, reason="policy denied")

    registry.register(HookPoint.BEFORE_MEMORY_WRITE, _block)
    view = _view(hook_registry=registry)

    with pytest.raises(MemoryViewAccessDenied, match="policy denied"):
        await view.write("findings", "vendor.a", {"score": 1})

    loaded = await view.read("findings", "vendor.a")
    assert loaded is None


@pytest.mark.asyncio
async def test_before_memory_write_hook_can_mutate_value() -> None:
    registry = HookRegistry()

    def _mutate(ctx: HookContext) -> HookResult:
        payload = dict(ctx.runtime_state.get("memory_write") or {})
        value = dict(payload.get("value") or {})
        value["score"] = 99
        value["mutated"] = True
        return HookResult(
            action=HookAction.MODIFY,
            modified_payload={"value": value},
        )

    registry.register(HookPoint.BEFORE_MEMORY_WRITE, _mutate)
    view = _view(hook_registry=registry)

    await view.write("findings", "vendor.a", {"score": 1})
    loaded = await view.read("findings", "vendor.a")

    assert loaded == {"score": 99, "mutated": True}
