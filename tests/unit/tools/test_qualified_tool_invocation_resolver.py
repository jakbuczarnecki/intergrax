# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

import pytest

from intergrax.contracts.execution_identity import TaskId
from intergrax.contracts.tool_execution_outcome import ToolExecutionResult
from intergrax.contracts.tools.tool_invocation_wiring_resolution import (
    ToolInvocationContext,
    ToolInvocationWiringResolver,
)
from intergrax.tools.execution_models import ToolExecutionResult as ToolsToolExecutionResult
from intergrax.tools.invocation_wiring import (
    DelegatingToolInvocationWiringResolver,
    ToolInvocationWiring,
    ToolRegistrationWiringView,
)
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


class _PassthroughWiringResolver:
    def resolve(
        self,
        *,
        tool_id: str,
        invocation_context: ToolInvocationContext,
        registration_wiring: ToolRegistrationWiringView,
    ) -> ToolInvocationWiring:
        _ = tool_id, invocation_context, registration_wiring
        return ToolInvocationWiring.empty()


def test_resolve_preserves_injected_wiring_resolver() -> None:
    wiring_resolver: ToolInvocationWiringResolver = _PassthroughWiringResolver()
    resolver = DefaultQualifiedToolInvocationResolver()
    request = resolver.resolve(
        activated_tool_id="tool-1",
        selected_operation="invoke",
        material=_Material(value="x"),
        tenant_id="tenant-a",
        task_id=TaskId("task_00000000000000000000000000000001"),
        run_id="run-1",
        agent_id="agent-caller",
        step_id="qmte:exec-req-3",
        execution_request_id="exec-req-3",
        correlation_request_id=None,
        idempotency_key=None,
        wiring_resolver=wiring_resolver,
    )
    assert request.wiring_resolver is wiring_resolver
    assert DelegatingToolInvocationWiringResolver().resolve(
        tool_id="tool-1",
        invocation_context=ToolInvocationContext(
            run_id="run-1",
            step_id="step-1",
            tool_id="tool-1",
            wiring_resolver=wiring_resolver,
        ),
        registration_wiring=ToolRegistrationWiringView(),
    ) == ToolInvocationWiring.empty()


def test_tool_execution_result_single_canonical_definition() -> None:
    assert ToolExecutionResult is ToolsToolExecutionResult
