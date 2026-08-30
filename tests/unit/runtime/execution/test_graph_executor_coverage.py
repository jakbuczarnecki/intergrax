# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id, mint_task_id
from intergrax.runtime.long_running.checkpoint_builder import should_skip_graph_node
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionCheckpointEntry,
    ExecutionCheckpointStatus,
    ExecutionPriorOutput,
    ExecutionTreeSnapshot,
)
from intergrax.runtime.long_running.runtime_checkpoint import RuntimeCheckpoint
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode, ExecutionNodeStatus

pytestmark = pytest.mark.gate


def _runtime_with_completed_child() -> RuntimeCheckpoint:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root = mint_execution_id()
    child = mint_execution_id()
    return RuntimeCheckpoint(
        run_id=run_id,
        attempt_id=attempt_id,
        execution_tree=ExecutionTreeSnapshot(
            task_id=task_id,
            run_id=run_id,
            attempt_id=attempt_id,
            entries=[
                ExecutionCheckpointEntry(
                    execution_id=root,
                    parent_execution_id=None,
                    status=ExecutionCheckpointStatus.RUNNING,
                ),
                ExecutionCheckpointEntry(
                    execution_id=child,
                    parent_execution_id=root,
                    status=ExecutionCheckpointStatus.COMPLETED,
                    graph_node_id="n1",
                    prior_output=ExecutionPriorOutput(
                        agent_id="a1",
                        summary="done",
                        status="completed",
                        graph_node_id="n1",
                    ),
                ),
            ],
        ),
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
    )


def test_should_skip_completed_node_when_prior_output_exists() -> None:
    node = ExecutionNode(node_id="n1", capability="cap.a", status=ExecutionNodeStatus.COMPLETED)
    prior = {
        "n1": AgentExecutionResult(
            agent_id="a1",
            run_id=mint_run_id(),
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    }
    checkpoint = _runtime_with_completed_child()
    assert should_skip_graph_node(node, checkpoint=checkpoint, prior_outputs=prior) is True


def test_should_not_skip_when_no_checkpoint() -> None:
    node = ExecutionNode(node_id="n2", capability="cap.b", status=ExecutionNodeStatus.PENDING)
    assert should_skip_graph_node(node, checkpoint=None, prior_outputs={}) is False
