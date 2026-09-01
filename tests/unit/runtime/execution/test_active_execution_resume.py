# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)
from intergrax.runtime.execution.active_execution_resume import (
    ActiveExecutionResumePlan,
    bind_active_execution_resume_plan,
    peek_active_execution_resume_plan,
    reset_active_execution_resume_plan,
)
from intergrax.runtime.long_running.execution_tree_checkpoint import (
    ExecutionTreeResumePlan,
    minimal_execution_tree_snapshot,
)

pytestmark = [pytest.mark.unit]


def _minimal_resume_plan() -> ExecutionTreeResumePlan:
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    root_execution_id = mint_execution_id()
    snapshot = minimal_execution_tree_snapshot(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        root_execution_id=root_execution_id,
    )
    return ExecutionTreeResumePlan(
        historical_snapshot=snapshot,
        active_snapshot=snapshot,
        historical_by_graph_node_id={},
    )


def test_bind_peek_reset_active_execution_resume_plan() -> None:
    assert peek_active_execution_resume_plan() is None
    plan = _minimal_resume_plan()
    token = bind_active_execution_resume_plan(ActiveExecutionResumePlan(plan=plan))
    try:
        active = peek_active_execution_resume_plan()
        assert active is not None
        assert active.plan is plan
    finally:
        reset_active_execution_resume_plan(token)
    assert peek_active_execution_resume_plan() is None


def test_reset_active_execution_resume_plan_after_exception() -> None:
    plan = _minimal_resume_plan()
    token = bind_active_execution_resume_plan(ActiveExecutionResumePlan(plan=plan))
    with pytest.raises(RuntimeError, match="boom"):
        try:
            raise RuntimeError("boom")
        finally:
            reset_active_execution_resume_plan(token)
    assert peek_active_execution_resume_plan() is None
