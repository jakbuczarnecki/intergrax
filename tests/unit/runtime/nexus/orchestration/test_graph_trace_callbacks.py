# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.nexus.orchestration.graph_trace_callbacks import GraphTraceCallbacks
from intergrax.runtime.nexus.retry.retry_engine import RetryRecord
from intergrax.runtime.task.task import Task

pytestmark = pytest.mark.gate


def test_graph_trace_callbacks_emit_node_and_retry_messages() -> None:
    emitter = MagicMock()
    task = Task(tenant_id="t1", user_id="u1", agent_id="a1", message="hi")
    callbacks = GraphTraceCallbacks(task=task, trace_emitter=emitter)

    node = ExecutionNode(node_id="n1", capability="cap.a", status=ExecutionNodeStatus.RUNNING)
    callbacks.on_node_start(node)
    node.status = ExecutionNodeStatus.COMPLETED
    callbacks.on_node_complete(node)
    callbacks.on_retry(
        RetryRecord(
            attempt=1,
            agent_id="a1",
            reason="validation",
            alternate_agent_id="alt",
        )
    )

    assert emitter.emit.call_count == 1
    assert emitter.emit_trace_step.call_count == 2
    start_call = emitter.emit_trace_step.call_args_list[0]
    assert start_call.kwargs["step"] == "graph.node_start"
    assert start_call.kwargs["payload"].node_id == "n1"
