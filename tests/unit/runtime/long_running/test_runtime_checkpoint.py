# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.contracts.agent_execution_result import AgentExecutionResult, AgentExecutionStatus
from intergrax.runtime.long_running.checkpoint_builder import (
    build_runtime_checkpoint,
    should_skip_uaep_step,
)
from intergrax.runtime.long_running.runtime_checkpoint import (
    RUNTIME_CHECKPOINT_KEY,
    RuntimeCheckpoint,
    runtime_checkpoint_from_execution_structured,
)
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
