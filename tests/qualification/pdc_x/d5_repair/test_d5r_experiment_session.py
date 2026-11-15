# © Artur Czarnecki. All rights reserved.

"""T-IGX-041-E (F-D5-07): ExperimentSession.run resolves ``mint_run_id`` and binds a governance identity."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.experiments import workflow
from intergrax.experiments.models import RegisterExperimentRequest
from intergrax.experiments.workflow import ExperimentSession
from intergrax.runtime.execution.certified_internal_harness_governance_identity import (
    admit_certified_internal_harness_root_governance_identity,
)
from intergrax.runtime.task.task import Task, TaskState
from testing_support.agent_registry_bootstrap import build_harness_registry

pytestmark = [pytest.mark.unit, pytest.mark.no_ci]


def test_workflow_module_resolves_mint_run_id() -> None:
    assert callable(workflow.mint_run_id)


async def _run(tmp_path: Path, *, user_id: str):
    session = ExperimentSession(
        experiments_db=tmp_path / "experiments.db",
        trace_db=tmp_path / "trace.db",
        tenant_id="t1",
        user_id=user_id,
    )
    record = session.register(
        RegisterExperimentRequest(
            hypothesis="Echo returns prefixed answer",
            capability="echo.basic",
            agent_id="echo",
            expected_output="hello workflow",
            validation_criteria="non-empty answer",
        )
    )
    loop = session.build_nexus_loop(build_harness_registry())
    return await session.run(loop=loop, record=record, message="hello workflow")


@pytest.mark.asyncio
async def test_session_run_completes_with_a_bound_governance_identity(tmp_path: Path) -> None:
    outcome = await _run(tmp_path, user_id="u1")
    assert outcome.task_result.state == TaskState.COMPLETED
    assert "pre_model governance identity unavailable" not in outcome.task_result.answer


@pytest.mark.asyncio
async def test_session_run_with_a_blank_user_never_completes(tmp_path: Path) -> None:
    try:
        outcome = await _run(tmp_path, user_id="")
    except (ValueError, RuntimeError):
        return
    assert outcome.task_result.state != TaskState.COMPLETED
    assert not outcome.passed


def test_admission_hook_refuses_a_blank_user() -> None:
    task = Task(tenant_id="t1", user_id="", message="x")
    with pytest.raises(ValueError, match="user_id"):
        admit_certified_internal_harness_root_governance_identity(task)
    ok = Task(tenant_id="t1", user_id="u1", message="x")
    assert admit_certified_internal_harness_root_governance_identity(ok).principal_id == "u1"
