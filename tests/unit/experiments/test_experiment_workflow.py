# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.experiments.models import ExperimentDecision, RegisterExperimentRequest
from intergrax.experiments.workflow import (
    ExperimentSession,
    evaluate_against_criteria,
    ensure_repo_root_on_path,
)
from testing_support.agent_registry_bootstrap import build_harness_registry
from intergrax.runtime.task.task import TaskResult, TaskState

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.no_ci]


def test_ensure_repo_root_on_path(tmp_path):
    (tmp_path / "intergrax").mkdir()
    (tmp_path / "agents").mkdir()
    root = ensure_repo_root_on_path(tmp_path)
    assert root == tmp_path.resolve()


@pytest.mark.asyncio
async def test_experiment_session_echo_workflow(tmp_path):
    experiments_db = tmp_path / "experiments.db"
    trace_db = tmp_path / "trace.db"

    session = ExperimentSession(
        experiments_db=experiments_db,
        trace_db=trace_db,
        tenant_id="t1",
        user_id="u1",
    )
    record = session.register(
        RegisterExperimentRequest(
            hypothesis="Echo returns prefixed answer",
            capability="echo.basic",
            agent_id="echo",
            expected_output="hello workflow",
            validation_criteria="non-empty answer with echo prefix",
        )
    )

    loop = session.build_nexus_loop(build_harness_registry())
    outcome = await session.run(
        loop=loop,
        record=record,
        message="hello workflow",
    )

    assert outcome.passed
    assert outcome.task_result.state == TaskState.COMPLETED
    assert "hello workflow" in outcome.task_result.answer
    assert outcome.record.run_ids
    assert outcome.trace_event_count > 0

    summary = session.summarize_trace(outcome.task_result.run_id or outcome.task_result.task_id)
    assert summary["event_count"] > 0

    decided = session.decide(
        record.experiment_id,
        ExperimentDecision.KEEP,
        notes="gate test",
    )
    assert decided.decision == ExperimentDecision.KEEP


def test_evaluate_against_criteria_expected_output():
    from intergrax.experiments.models import ExperimentRecord

    record = ExperimentRecord(
        experiment_id="exp1",
        hypothesis="test",
        capability="echo.basic",
        expected_output="needle",
        created_at_utc="2026-01-01T00:00:00Z",
        updated_at_utc="2026-01-01T00:00:00Z",
    )
    result = TaskResult(
        task_id="task1",
        run_id="run1",
        state=TaskState.COMPLETED,
        answer="echo: needle in haystack",
        metadata={"validation_valid": True},
    )
    checks = evaluate_against_criteria(record, result)
    assert checks["expected_output_substring"] is True
    assert checks["completed"] is True
