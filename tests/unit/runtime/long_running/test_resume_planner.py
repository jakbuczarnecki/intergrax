# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.agent_decision import AgentDecisionType
from intergrax.runtime.human.models import HumanResponseVerdict
from intergrax.runtime.long_running.execution_tree_checkpoint import minimal_runtime_checkpoint
from intergrax.runtime.long_running.models import TaskCheckpoint
from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id, mint_run_id
from intergrax.runtime.long_running.resume_planner import (
    build_timeout_resume_task,
    timeout_action_to_verdict,
)
from intergrax.runtime.task.task import Task, TaskContext, TaskState
from intergrax.runtime.task.task_contract import TaskExecutionOptions, TaskLongRunningOptions

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_timeout_action_to_verdict_mapping() -> None:
    assert timeout_action_to_verdict(AgentDecisionType.ESCALATE) == HumanResponseVerdict.ESCALATE
    assert timeout_action_to_verdict(AgentDecisionType.FAIL) == HumanResponseVerdict.REJECT
    assert timeout_action_to_verdict(AgentDecisionType.CANCEL) == HumanResponseVerdict.REJECT


def test_build_timeout_resume_task_sets_scheduler_metadata() -> None:
    task = Task(
        tenant_id="t1",
        user_id="u1",
        message="monitor",
        context=TaskContext(capability="hitl.basic"),
        options=TaskExecutionOptions(
            long_running=TaskLongRunningOptions(enabled=True),
        ),
    )
    checkpoint = TaskCheckpoint(
        task_id=task.task_id,
        tenant_id="t1",
        resume_token="rt_test",
        task_state=TaskState.WAITING_FOR_HUMAN,
        task_snapshot=task.model_dump(mode="json"),
        runtime=minimal_runtime_checkpoint(
            task_id=task.task_id,
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            root_execution_id=mint_execution_id(),
        ),
    )
    resume = build_timeout_resume_task(
        checkpoint,
        verdict=HumanResponseVerdict.REJECT,
        action=AgentDecisionType.FAIL,
        target_task_id=task.task_id,
        target_tenant_id="t1",
    )
    assert resume.options.long_running.resume_token == "rt_test"
    assert resume.options.human.verdict == "reject"
    assert resume.metadata["scheduler_timeout"] is True
    assert resume.metadata["scheduler_timeout_action"] == "fail"
