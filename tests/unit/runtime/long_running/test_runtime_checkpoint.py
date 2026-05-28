# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.long_running.checkpoint_builder import (
    apply_runtime_checkpoint_to_graph,
    build_runtime_checkpoint,
    should_skip_graph_node,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    RuntimeCheckpoint,
    runtime_checkpoint_from_execution_structured,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionGraph, ExecutionNode, ExecutionNodeStatus
from intergrax.runtime.task.task import Task


@pytest.mark.unit
@pytest.mark.gate
def test_runtime_checkpoint_from_execution_structured():
    structured = {
        RUNTIME_CHECKPOINT_KEY: {
            "uaep_step_index": 1,
            "uaep_step_id": "review",
            "last_step_output": {"step_id": "review", "summary": "done"},
        }
    }
    ckpt = runtime_checkpoint_from_execution_structured(structured)
    assert ckpt is not None
    assert ckpt.uaep_step_index == 1
    assert ckpt.uaep_step_id == "review"


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_uaep_step_when_resumed_at_same_index():
    ckpt = RuntimeCheckpoint(
        uaep_step_index=0,
        uaep_step_id="review",
        last_step_output={"step_id": "review", "summary": "pending"},
    )
    assert should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        human_approved=True,
    )
    assert not should_skip_uaep_step(
        step_index=0,
        step_id="review",
        checkpoint=ckpt,
        human_approved=False,
    )
    assert not should_skip_uaep_step(
        step_index=1,
        step_id="review",
        checkpoint=ckpt,
        human_approved=True,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_build_runtime_checkpoint_merges_execution_structured():
    task = Task(tenant_id="t1", user_id="u1", agent_id="hitl")
    execution = AgentExecutionResult(
        agent_id="hitl",
        run_id="run_1",
        status=AgentExecutionStatus.NEEDS_INPUT,
        summary="awaiting approval",
        structured_data={
            RUNTIME_CHECKPOINT_KEY: {
                "uaep_step_index": 0,
                "uaep_step_id": "review",
                "last_step_output": {"step_id": "review", "summary": "pending review"},
            }
        },
    )
    runtime = build_runtime_checkpoint(task, last_execution=execution)
    assert runtime.uaep_step_index == 0
    assert runtime.uaep_step_id == "review"
    assert runtime.last_step_output is not None


@pytest.mark.unit
@pytest.mark.gate
def test_should_skip_graph_node_when_completed_with_prior_output():
    ckpt = RuntimeCheckpoint(
        node_states={"n1": ExecutionNodeStatus.COMPLETED.value},
        prior_node_outputs={"n1": {"agent_id": "a1", "summary": "done", "status": "completed"}},
    )
    node = ExecutionNode(node_id="n1", agent_id="a1", status=ExecutionNodeStatus.COMPLETED)
    prior = {
        "n1": AgentExecutionResult(
            agent_id="a1",
            run_id="run_1",
            status=AgentExecutionStatus.COMPLETED,
            summary="done",
        )
    }
    assert should_skip_graph_node(node, checkpoint=ckpt, prior_outputs=prior)
    failed = ExecutionNode(node_id="n2", agent_id="a2", status=ExecutionNodeStatus.FAILED)
    assert not should_skip_graph_node(failed, checkpoint=ckpt, prior_outputs=prior)


@pytest.mark.unit
@pytest.mark.gate
def test_apply_runtime_checkpoint_to_graph_restores_completed_node():
    graph = ExecutionGraph(
        graph_id="g1",
        task_id="task_1",
        nodes=[
            ExecutionNode(node_id="n1", agent_id="a1"),
            ExecutionNode(node_id="n2", agent_id="a2", depends_on=["n1"]),
        ],
    )
    runtime = RuntimeCheckpoint(
        node_states={
            "n1": ExecutionNodeStatus.COMPLETED.value,
            "n2": ExecutionNodeStatus.FAILED.value,
        },
        prior_node_outputs={
            "n1": {"agent_id": "a1", "summary": "step one", "status": "completed"},
        },
    )
    prior: dict = {}
    apply_runtime_checkpoint_to_graph(graph, runtime, prior)

    assert graph.node_by_id("n1").status == ExecutionNodeStatus.COMPLETED
    assert graph.node_by_id("n1").execution_result is not None
    assert graph.node_by_id("n1").execution_result.summary == "step one"
    assert graph.node_by_id("n2").status == ExecutionNodeStatus.FAILED
    assert "n1" in prior
