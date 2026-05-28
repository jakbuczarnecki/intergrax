# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.nexus.artifacts.models import ArtifactRef
from intergrax.runtime.nexus.context.context_manager import ContextManager
from intergrax.runtime.nexus.context.shared_task_context import (
    SharedContextConflictError,
    SharedTaskContext,
    load_shared_task_context,
)
from intergrax.runtime.nexus.execution.execution_graph import ExecutionNode
from intergrax.runtime.task.task import Task


def _task() -> Task:
    return Task(tenant_id="t1", user_id="u1", message="hello", task_id="task_shared_1")


def _execution(*, agent_id: str = "agent_a", summary: str = "done") -> AgentExecutionResult:
    return AgentExecutionResult(
        agent_id=agent_id,
        run_id="task_shared_1",
        status=AgentExecutionStatus.COMPLETED,
        summary=summary,
        structured_data={"score": 9},
    )


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_creates_shared_context_on_task():
    manager = ContextManager()
    task = _task()
    shared = manager.ensure_shared_context(task)
    assert shared.task_id == task.task_id
    assert shared.version == 1
    assert load_shared_task_context(task) is not None


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_record_node_output_increments_version():
    manager = ContextManager()
    task = _task()
    node = ExecutionNode(node_id="n1", agent_id="agent_a", capability="cap.a")
    execution = _execution(summary="first result")

    shared = manager.record_node_output(task, node, execution)

    assert shared.version == 2
    assert shared.structured_outputs["n1"]["summary"] == "first result"
    assert shared.structured_outputs["n1"]["structured_data"]["score"] == 9
    reloaded = load_shared_task_context(task)
    assert reloaded is not None
    assert reloaded.structured_outputs["n1"]["agent_id"] == "agent_a"


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_apply_to_task_injects_shared_context():
    manager = ContextManager()
    task = _task()
    node = ExecutionNode(node_id="n2", agent_id="agent_b", capability="cap.b", depends_on=["n1"])
    prior = {
        "n1": _execution(summary="prior output"),
    }
    manager.record_node_output(task, ExecutionNode(node_id="n1", agent_id="agent_a"), prior["n1"])
    bundle = manager.build_agent_context(task, node, prior)
    node_task = manager.apply_to_task(task, bundle)

    assert "prior output" in node_task.message
    shared = load_shared_task_context(node_task)
    assert shared is not None
    assert "n1" in shared.structured_outputs


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_put_structured_output_conflict():
    manager = ContextManager()
    task = _task()
    manager.ensure_shared_context(task)
    with pytest.raises(SharedContextConflictError, match="version mismatch"):
        manager.put_structured_output(
            task,
            key="handoff",
            payload={"step": 1},
            expected_version=99,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_context_manager_put_artifact():
    manager = ContextManager()
    task = _task()
    ref = ArtifactRef(artifact_id="art_1", kind="markdown", size_bytes=128)
    shared = manager.put_artifact(task, label="product_spec.md", artifact=ref)
    assert shared.artifacts["product_spec.md"].artifact_id == "art_1"
    assert isinstance(load_shared_task_context(task), SharedTaskContext)


@pytest.mark.unit
@pytest.mark.gate
def test_shared_task_context_roundtrip_json():
    shared = SharedTaskContext(
        task_id="task_1",
        structured_outputs={"n1": {"summary": "ok"}},
        version=3,
    )
    restored = SharedTaskContext.model_validate(shared.model_dump(mode="json"))
    assert restored.version == 3
    assert restored.structured_outputs["n1"]["summary"] == "ok"
