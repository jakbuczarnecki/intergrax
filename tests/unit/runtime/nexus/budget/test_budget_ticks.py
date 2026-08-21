# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import pytest

from intergrax.runtime.nexus.budget.budget_enforcer import BudgetExceededError
from intergrax.runtime.nexus.budget.budget_models import BudgetEnforcementMode, BudgetPolicy, RunBudget
from intergrax.runtime.nexus.budget.budget_ticks import (
    enforce_tool_call_budget,
    enforce_wall_time_budget,
    record_planner_iteration_and_enforce,
    record_rag_invocation_and_enforce,
    record_websearch_invocation_and_enforce,
    run_elapsed_seconds,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState, ToolCallTrace
from testing_support.builder import build_runtime_state_for_tests

pytestmark = pytest.mark.unit


def _state_with_budget(*, max_rag: int | None = None, max_web: int | None = None) -> RuntimeState:
    st = build_runtime_state_for_tests(run_id="budget-tick")
    st.context.config.run_budget = RunBudget(
        max_rag_invocations=max_rag,
        max_websearch_invocations=max_web,
        max_tool_calls=2,
    )
    st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    return st


def test_rag_second_invocation_aborts_when_limit_one() -> None:
    st = _state_with_budget(max_rag=1)
    record_rag_invocation_and_enforce(st)
    assert st.rag_step_invocation_count == 1
    with pytest.raises(BudgetExceededError, match="max_rag_invocations"):
        record_rag_invocation_and_enforce(st)


def test_websearch_second_invocation_aborts_when_limit_one() -> None:
    st = _state_with_budget(max_web=1)
    record_websearch_invocation_and_enforce(st)
    with pytest.raises(BudgetExceededError, match="max_websearch_invocations"):
        record_websearch_invocation_and_enforce(st)


def test_tool_budget_enforce_after_traces() -> None:
    st = build_runtime_state_for_tests(run_id="tool-budget")
    st.context.config.run_budget = RunBudget(max_tool_calls=1)
    st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    fake = ToolCallTrace(
        tool_name="t",
        arguments={},
        output_preview=None,
        success=True,
        error_message=None,
        raw_trace={},
    )
    st.tool_traces.append(fake)
    enforce_tool_call_budget(st)
    st.tool_traces.append(fake)
    with pytest.raises(BudgetExceededError, match="max_tool_calls"):
        enforce_tool_call_budget(st)


def test_no_budget_config_no_op() -> None:
    st = build_runtime_state_for_tests(run_id="no-budget")
    st.context.config.run_budget = None
    st.context.config.budget_policy = None
    record_rag_invocation_and_enforce(st)
    record_rag_invocation_and_enforce(st)
    assert st.rag_step_invocation_count == 2


def test_planner_iteration_second_round_aborts_when_limit_one() -> None:
    st = build_runtime_state_for_tests(run_id="planner-budget")
    st.context.config.run_budget = RunBudget(max_planner_iterations=1)
    st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    record_planner_iteration_and_enforce(st)
    assert st.planner_iteration_count == 1
    with pytest.raises(BudgetExceededError, match="max_planner_iterations"):
        record_planner_iteration_and_enforce(st)


def test_wall_time_budget_aborts_when_elapsed_exceeds_limit() -> None:
    from datetime import datetime, timedelta, timezone

    st = build_runtime_state_for_tests(run_id="wall-budget")
    st.context.config.run_budget = RunBudget(max_wall_time_seconds=1.0)
    st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
    st.started_at_utc = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
    assert run_elapsed_seconds(st) > 1.0
    with pytest.raises(BudgetExceededError, match="max_wall_time_seconds"):
        enforce_wall_time_budget(st)
