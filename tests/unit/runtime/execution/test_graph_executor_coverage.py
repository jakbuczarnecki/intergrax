# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.long_running.checkpoint_builder import should_skip_graph_node
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode, ExecutionNodeStatus

pytestmark = pytest.mark.gate


def test_should_skip_completed_node_when_prior_output_exists() -> None:
    node = ExecutionNode(node_id="n1", capability="cap.a", status=ExecutionNodeStatus.COMPLETED)
    prior = {
        "n1": AgentExecutionResult(
            agent_id="a1",
            run_id="r1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    }
    checkpoint = RuntimeCheckpoint(
        checkpoint_id="c1",
        task_id="t1",
        resume_token="tok",
        progress_message="",
    )
    assert should_skip_graph_node(node, checkpoint=checkpoint, prior_outputs=prior) is True


def test_should_not_skip_when_no_checkpoint() -> None:
    node = ExecutionNode(node_id="n2", capability="cap.b", status=ExecutionNodeStatus.PENDING)
    assert should_skip_graph_node(node, checkpoint=None, prior_outputs={}) is False
