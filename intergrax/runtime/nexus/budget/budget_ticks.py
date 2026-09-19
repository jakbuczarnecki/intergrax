# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""
Mid-run :class:`RunBudget` enforcement for tool-context invocations (RAG, websearch, tools).

Called from :mod:`intergrax.runtime.nexus.tools.plan_context_invocation` and
:mod:`intergrax.runtime.nexus.tools.tool_runtime` so usage-based limits apply
before tool loops finish.
"""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.execution_deadline.clock import MonotonicClockPort
from intergrax.contracts.execution_deadline.projection import ExecutionDeadlineProjection
from intergrax.runtime.execution.active_execution_budget import peek_active_execution_budget
from intergrax.runtime.execution.budget.consumption import (
    consume_planner_iteration,
    consume_rag_invocation,
    consume_replan,
    consume_tool_call,
    consume_wall_time_delta,
    consume_websearch_invocation,
)
from intergrax.runtime.execution.deadline_authority.system_clocks import SystemMonotonicClock
from intergrax.runtime.execution.deadline_scope import (
    peek_active_execution_deadline_projection,
    peek_active_execution_monotonic_clock,
)
from intergrax.runtime.execution.live_deadline_evaluator import execution_is_expired_now
from intergrax.runtime.nexus.budget.budget_enforcer import BudgetEnforcer, BudgetExceededError
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.utils.time_provider import SystemTimeProvider


def _enforcer(state: RuntimeState) -> BudgetEnforcer | None:
    cfg = state.context.config
    if cfg.run_budget is None or cfg.budget_policy is None:
        return None
    return BudgetEnforcer(cfg.run_budget, cfg.budget_policy)


def record_rag_invocation_and_enforce(state: RuntimeState) -> None:
    """After confirming RAG is enabled and configured; before retrieval work."""
    consume_rag_invocation()
    state.rag_step_invocation_count += 1
    enc = _enforcer(state)
    if enc is not None:
        enc.check_rag_invocations(
            run_id=state.run_id,
            rag_invocations=state.rag_step_invocation_count,
            state=state,
        )


def record_websearch_invocation_and_enforce(state: RuntimeState) -> None:
    """After confirming websearch is enabled and configured; before network/search work."""
    consume_websearch_invocation()
    state.websearch_step_invocation_count += 1
    enc = _enforcer(state)
    if enc is not None:
        enc.check_websearch_invocations(
            run_id=state.run_id,
            websearch_invocations=state.websearch_step_invocation_count,
            state=state,
        )


def record_tool_call_and_enforce(state: RuntimeState) -> None:
    """Before each real tool invocation is committed."""
    consume_tool_call()
    enc = _enforcer(state)
    if enc is not None:
        projected = len(state.tool_traces) + 1
        enc.check_tool_calls(
            run_id=state.run_id,
            tool_calls=projected,
            state=state,
        )


def enforce_tool_call_budget(state: RuntimeState) -> None:
    """After each tool trace is appended (legacy cumulative ``BudgetEnforcer`` check)."""
    enc = _enforcer(state)
    if enc is not None:
        enc.check_tool_calls(
            run_id=state.run_id,
            tool_calls=len(state.tool_traces),
            state=state,
        )


def run_elapsed_seconds(state: RuntimeState) -> float:
    """Observability elapsed since ``RuntimeState.started_at_utc`` (not wall-time authority)."""
    started = datetime.fromisoformat(state.started_at_utc)
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (SystemTimeProvider.utc_now() - started).total_seconds()


def _wall_checkpoint_projection_and_clock(
    state: RuntimeState,
) -> tuple[ExecutionDeadlineProjection, MonotonicClockPort] | None:
    """Resolve active canonical monotonic authority for mid-loop checkpoints."""
    projection = peek_active_execution_deadline_projection()
    monotonic_clock = peek_active_execution_monotonic_clock()
    if projection is not None:
        if projection.global_deadline_monotonic is None:
            return None
        if monotonic_clock is None:
            raise RuntimeError(
                "active execution monotonic clock required for bounded wall-time checkpoint",
            )
        return projection, monotonic_clock

    budget_state = peek_active_execution_budget()
    if budget_state is not None and budget_state.global_deadline_monotonic is not None:
        clock = SystemMonotonicClock()
        bound = budget_state.global_deadline_monotonic
        remaining = max(0.0, bound - clock.monotonic())
        ephemeral = ExecutionDeadlineProjection(
            deadline_at_utc=None,
            remaining_seconds=remaining,
            is_expired=remaining <= 0.0,
            global_deadline_monotonic=bound,
        )
        return ephemeral, clock

    run_budget = state.context.config.run_budget
    if run_budget is not None and run_budget.max_wall_time_seconds is not None:
        raise RuntimeError(
            "canonical execution deadline projection required for bounded wall-time checkpoint",
        )
    return None


def _canonical_global_wall_time_expired(state: RuntimeState) -> bool | None:
    """``True`` expired, ``False`` available, ``None`` when globally unbounded."""
    resolved = _wall_checkpoint_projection_and_clock(state)
    if resolved is None:
        return None
    projection, monotonic_clock = resolved
    return execution_is_expired_now(projection, monotonic_clock)


def record_planner_iteration_and_enforce(state: RuntimeState) -> None:
    """Before each bounded ReAct planner round; increments run-level planner iteration count."""
    consume_planner_iteration()
    state.planner_iteration_count += 1
    enc = _enforcer(state)
    if enc is not None:
        enc.check_planner_iterations(
            run_id=state.run_id,
            planner_iterations=state.planner_iteration_count,
            state=state,
        )


def record_replan_and_enforce(state: RuntimeState) -> None:
    """When a real replan is committed for the active orchestration run."""
    consume_replan()
    state.replan_count += 1
    enc = _enforcer(state)
    if enc is not None:
        enc.check_replans(
            run_id=state.run_id,
            replans=state.replan_count,
            state=state,
        )


def enforce_wall_time_budget(state: RuntimeState) -> None:
    """Mid-run cooperative checkpoint delegating global wall-time to canonical authority."""
    elapsed = run_elapsed_seconds(state)
    consume_wall_time_delta(elapsed)
    expired = _canonical_global_wall_time_expired(state)
    if expired is None or not expired:
        return
    enc = _enforcer(state)
    if enc is not None:
        enc.check_wall_time(
            run_id=state.run_id,
            elapsed_seconds=elapsed,
            state=state,
        )
        return
    raise BudgetExceededError("Budget exceeded: max_wall_time_seconds (canonical deadline expired)")
