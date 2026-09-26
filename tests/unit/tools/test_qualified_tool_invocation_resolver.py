# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

import pytest

from intergrax.contracts.execution_identity import TaskId
from intergrax.tools.qualified_tool_invocation_resolver import DefaultQualifiedToolInvocationResolver

pytestmark = pytest.mark.unit


class _Material(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    value: str


def test_resolve_maps_canonical_fields() -> None:
    resolver = DefaultQualifiedToolInvocationResolver()
    request = resolver.resolve(
        activated_tool_id="tool-1",
        selected_operation="invoke",
        material=_Material(value="x"),
        tenant_id="tenant-a",
        task_id=TaskId("task_00000000000000000000000000000001"),
        run_id="run-1",
        agent_id="agent-caller",
        step_id="qmte:exec-req-1",
        execution_request_id="exec-req-1",
        correlation_request_id="execution-id-1",
        idempotency_key="qmte:exec-req-1:invoke",
    )
    assert request.tool_id == "tool-1"
    assert request.tenant_id == "tenant-a"
    assert request.task_id == "task_00000000000000000000000000000001"
    assert request.run_id == "run-1"
    assert request.agent_id == "agent-caller"
    assert request.step_id == "qmte:exec-req-1"
    assert request.correlation_request_id == "execution-id-1"
    assert request.idempotency_key == "qmte:exec-req-1:invoke"
    assert isinstance(request.input, _Material)


def test_default_idempotency_uses_normalized_selected_operation_semantic_intent() -> None:
    resolver = DefaultQualifiedToolInvocationResolver()
    request = resolver.resolve(
        activated_tool_id="tool-1",
        selected_operation="invoke",
        material=_Material(value="x"),
        tenant_id="tenant-a",
        task_id=TaskId("task_00000000000000000000000000000001"),
        run_id="run-1",
        agent_id="agent-caller",
        step_id="qmte:exec-req-2",
        execution_request_id="exec-req-2",
        correlation_request_id=None,
        idempotency_key=None,
    )
    assert request.idempotency_key == "qmte:exec-req-2:invoke"
    assert request.tool_id == "tool-1"
