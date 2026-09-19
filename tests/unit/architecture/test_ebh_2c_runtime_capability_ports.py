# © Artur Czarnecki. All rights reserved.

"""EBH-2C — typed runtime capability ports on ``RuntimeExecutionContext``."""

from __future__ import annotations

import ast
import inspect
import types
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import pytest

from intergrax.contracts.execution_deadline import ExecutionCancellationView
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.contracts.memory_write_policy import MemoryWritePolicy
from intergrax.contracts.runtime_event import RuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.contracts.runtime_execution_context import (
    EventEmitter,
    MemoryView,
    MetadataCarrier,
    RuntimeExecutionContext,
    ToolGateway,
)
from intergrax.contracts.tool_request import ToolRequest, ToolResponse, ToolResponseStatus
from intergrax.runtime.cancellation.coordinator import CANCELLATION_REQUESTED_KEY
from intergrax.runtime.cancellation.runtime_execution_cancellation_view import (
    RuntimeExecutionContextCancellationView,
    attach_runtime_execution_cancellation_view,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CONTEXT_MODULE = _REPO_ROOT / "intergrax/contracts/runtime_execution_context.py"

_CAPABILITY_FIELD_NAMES = (
    "tool_gateway",
    "event_emitter",
    "memory_view",
    "request",
    "cancellation",
)

_FORBIDDEN_CAPABILITY_FIELDS = frozenset(
    {
        "trace",
        "domain_context",
    }
)


def _module_runtime_imports(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(
            "intergrax.runtime"
        ):
            hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime"):
                    hits.append(alias.name)
    return hits


def _capability_field_types() -> dict[str, object]:
    return get_type_hints(RuntimeExecutionContext, include_extras=True)


def _union_members(resolved: object) -> tuple[object, ...]:
    origin = get_origin(resolved)
    if origin is None:
        return (resolved,)
    if origin is types.UnionType or str(origin) == "typing.Union":
        return get_args(resolved)
    return (resolved,)


def test_capability_field_types_are_public_ports() -> None:
    expected: dict[str, type] = {
        "tool_gateway": ToolGateway,
        "event_emitter": EventEmitter,
        "memory_view": MemoryView,
        "request": MetadataCarrier,
        "cancellation": ExecutionCancellationView,
    }
    hints = _capability_field_types()
    for name, protocol in expected.items():
        resolved = hints[name]
        members = _union_members(resolved)
        assert type(None) in members or None in members
        non_none = [m for m in members if m is not type(None)]
        assert len(non_none) == 1
        assert non_none[0] is protocol


def test_no_capability_field_uses_any() -> None:
    hints = _capability_field_types()
    for name in _CAPABILITY_FIELD_NAMES:
        resolved = hints[name]
        for member in _union_members(resolved):
            assert member is not Any


def test_removed_legacy_capability_fields() -> None:
    for name in _FORBIDDEN_CAPABILITY_FIELDS:
        assert name not in RuntimeExecutionContext.model_fields


def test_runtime_execution_context_has_no_runtime_imports() -> None:
    assert _module_runtime_imports(_CONTEXT_MODULE) == []


def test_event_helpers_use_canonical_runtime_event_type() -> None:
    from intergrax.contracts import runtime_execution_context as rec

    assert rec.RuntimeEvent is RuntimeEvent
    assert rec.RuntimeEventType is RuntimeEventType
    sig = inspect.signature(rec._tool_requested_event_type)
    helper_hints = get_type_hints(rec._tool_requested_event_type)
    assert helper_hints["return"] is RuntimeEventType
    assert sig.return_annotation in (RuntimeEventType, "RuntimeEventType")


class _ToolGatewayPlugin:
    async def invoke(self, request: ToolRequest) -> ToolResponse:
        return ToolResponse(
            request_id=request.request_id,
            status=ToolResponseStatus.SUCCESS,
        )


class _EventEmitterPlugin:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = []

    async def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)


class _MemoryViewPlugin:
    async def read(self, namespace: str, key: str) -> dict[str, Any] | None:
        return None

    async def write(
        self,
        namespace: str,
        key: str,
        value: dict[str, Any],
        *,
        policy: MemoryWritePolicy = MemoryWritePolicy.REPLACE,
    ) -> None:
        return None

    async def list(self, namespace: str, prefix: str = "") -> list[Any]:
        return []


class _CancellationViewPlugin:
    def __init__(self, *, cancelled: bool) -> None:
        self._cancelled = cancelled

    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancellation_reason(self) -> str | None:
        return "operator" if self._cancelled else None


class _MetadataCarrierPlugin:
    metadata: dict[str, Any]


def _minimal_exec_ctx(**overrides: object) -> RuntimeExecutionContext:
    base = {
        "task_id": mint_task_id(),
        "run_id": mint_run_id(),
        "attempt_id": mint_attempt_id(),
        "execution_id": mint_execution_id(),
        "agent_id": "plugin.agent",
    }
    base.update(overrides)
    return RuntimeExecutionContext(**base)


def test_structural_capability_plugins_accepted_by_context() -> None:
    carrier = _MetadataCarrierPlugin()
    carrier.metadata = {}
    ctx = _minimal_exec_ctx(
        tool_gateway=_ToolGatewayPlugin(),
        event_emitter=_EventEmitterPlugin(),
        memory_view=_MemoryViewPlugin(),
        request=carrier,
        cancellation=_CancellationViewPlugin(cancelled=False),
    )
    assert isinstance(ctx.tool_gateway, ToolGateway)
    assert isinstance(ctx.event_emitter, EventEmitter)
    assert isinstance(ctx.memory_view, MemoryView)
    assert isinstance(ctx.request, MetadataCarrier)
    assert ctx.cancellation is not None
    assert ctx.cancellation.is_cancelled() is False


def test_should_cancel_without_port_returns_false() -> None:
    ctx = _minimal_exec_ctx()
    assert ctx.should_cancel() is False


def test_should_cancel_delegates_to_port() -> None:
    ctx = _minimal_exec_ctx(cancellation=_CancellationViewPlugin(cancelled=False))
    assert ctx.should_cancel() is False
    ctx.cancellation = _CancellationViewPlugin(cancelled=True)
    assert ctx.should_cancel() is True


def test_runtime_cancellation_adapter_matches_request_metadata() -> None:
    carrier = _MetadataCarrierPlugin()
    carrier.metadata = {CANCELLATION_REQUESTED_KEY: True}
    ctx = _minimal_exec_ctx(request=carrier)
    attach_runtime_execution_cancellation_view(ctx)
    assert ctx.should_cancel() is True


def test_runtime_cancellation_adapter_matches_execution_metadata() -> None:
    ctx = _minimal_exec_ctx(metadata={CANCELLATION_REQUESTED_KEY: True})
    attach_runtime_execution_cancellation_view(ctx)
    assert ctx.should_cancel() is True


def test_runtime_cancellation_adapter_is_execution_cancellation_view() -> None:
    ctx = _minimal_exec_ctx()
    attach_runtime_execution_cancellation_view(ctx)
    view = ctx.cancellation
    assert isinstance(view, RuntimeExecutionContextCancellationView)
    assert view.is_cancelled() is False
