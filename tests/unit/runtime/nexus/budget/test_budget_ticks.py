# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from datetime import datetime, timedelta, timezone

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
from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from testing_support.builder import governed_runtime_state_scope

pytestmark = pytest.mark.unit


def test_rag_second_invocation_aborts_when_limit_one() -> None:
    with governed_runtime_state_scope("budget-tick") as st:
        st.context.config.run_budget = RunBudget(max_rag_invocations=1)
        st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
        record_rag_invocation_and_enforce(st)
        assert st.rag_step_invocation_count == 1
        with pytest.raises(BudgetExceededError, match="max_rag_invocations"):
            record_rag_invocation_and_enforce(st)


def test_websearch_second_invocation_aborts_when_limit_one() -> None:
    with governed_runtime_state_scope("budget-tick") as st:
        st.context.config.run_budget = RunBudget(max_websearch_invocations=1)
        st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
        record_websearch_invocation_and_enforce(st)
        with pytest.raises(BudgetExceededError, match="max_websearch_invocations"):
            record_websearch_invocation_and_enforce(st)


def test_tool_budget_enforce_after_traces() -> None:
    with governed_runtime_state_scope("tool-budget") as st:
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
    with governed_runtime_state_scope("no-budget") as st:
        st.context.config.run_budget = None
        st.context.config.budget_policy = None
        record_rag_invocation_and_enforce(st)
        record_rag_invocation_and_enforce(st)
        assert st.rag_step_invocation_count == 2


def test_planner_iteration_second_round_aborts_when_limit_one() -> None:
    with governed_runtime_state_scope("planner-budget") as st:
        st.context.config.run_budget = RunBudget(max_planner_iterations=1)
        st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
        record_planner_iteration_and_enforce(st)
        assert st.planner_iteration_count == 1
        with pytest.raises(BudgetExceededError, match="max_planner_iterations"):
            record_planner_iteration_and_enforce(st)


def test_wall_time_budget_aborts_when_canonical_deadline_expired() -> None:
    from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
    from intergrax.runtime.execution.deadline_scope import (
        bind_active_execution_deadline_scope,
        reset_active_execution_deadline_scope,
    )
    from intergrax.runtime.execution.protected_work_admission import (
        CanonicalHardProtectedWorkAdmission,
        StaticCancellationView,
    )

    class _Clock:
        def __init__(self, value: float) -> None:
            self._value = value

        def monotonic(self) -> float:
            return self._value

    clock = _Clock(200.0)
    projection = ExecutionDeadlineProjection(
        deadline_at_utc=datetime(2020, 1, 1, tzinfo=timezone.utc),
        remaining_seconds=0.0,
        is_expired=True,
        global_deadline_monotonic=100.0,
    )
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=clock,
        ),
        monotonic_clock=clock,
    )
    try:
        with governed_runtime_state_scope("wall-budget") as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=3600.0)
            st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
            st.started_at_utc = datetime.now(timezone.utc).isoformat()
            with pytest.raises(BudgetExceededError, match="max_wall_time_seconds"):
                enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)


def test_wall_time_budget_unbounded_ignores_legacy_started_at() -> None:
    from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
    from intergrax.runtime.execution.deadline_scope import (
        bind_active_execution_deadline_scope,
        reset_active_execution_deadline_scope,
    )
    from intergrax.runtime.execution.protected_work_admission import (
        CanonicalHardProtectedWorkAdmission,
        StaticCancellationView,
    )

    class _Clock:
        def monotonic(self) -> float:
            return 1.0

    projection = ExecutionDeadlineProjection(
        deadline_at_utc=None,
        remaining_seconds=float("inf"),
        is_expired=False,
        global_deadline_monotonic=None,
    )
    clock = _Clock()
    tokens = bind_active_execution_deadline_scope(
        projection=projection,
        admission=CanonicalHardProtectedWorkAdmission(
            projection=projection,
            cancellation_view=StaticCancellationView(cancelled=False),
            monotonic_clock=clock,
        ),
        monotonic_clock=clock,
    )
    try:
        with governed_runtime_state_scope("wall-unbounded") as st:
            st.context.config.run_budget = RunBudget(max_wall_time_seconds=1.0)
            st.context.config.budget_policy = BudgetPolicy(enforcement_mode=BudgetEnforcementMode.ABORT)
            st.started_at_utc = (datetime.now(timezone.utc) - timedelta(seconds=5)).isoformat()
            enforce_wall_time_budget(st)
    finally:
        reset_active_execution_deadline_scope(*tokens)
