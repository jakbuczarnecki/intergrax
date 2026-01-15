# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.runtime.nexus.budget.budget_models import RunBudget, BudgetPolicy, BudgetEnforcementMode
from intergrax.runtime.nexus.budget.budget_diagnostics import BudgetExceededDiagV1
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


class BudgetExceededError(RuntimeError):
    """
    Raised when a hard budget is exceeded and enforcement mode is ABORT.
    """


class BudgetEnforcer:
    def __init__(self, budget: RunBudget, policy: BudgetPolicy) -> None:
        self._budget = budget
        self._policy = policy

    def check_llm_calls(self, *, run_id: str, llm_calls: int, state: RuntimeState) -> None:
        limit = self._budget.max_llm_calls
        if limit is None:
            return
        if llm_calls <= limit:
            return

        state.trace_event(
            component=TraceComponent.POLICY,
            step="budget_exceeded",
            level=TraceLevel.ERROR,
            message="Budget exceeded: max_llm_calls",
            payload=BudgetExceededDiagV1(
                run_id=run_id,
                budget_name="max_llm_calls",
                limit=limit,
                actual=llm_calls,
                enforcement_mode=self._policy.enforcement_mode.value,
            ),
        )

        if self._policy.enforcement_mode is BudgetEnforcementMode.ABORT:
            raise BudgetExceededError(f"Budget exceeded: max_llm_calls ({llm_calls} > {limit})")


    def check_tool_calls(self, *, run_id: str, tool_calls: int, state: RuntimeState) -> None:
        limit = self._budget.max_tool_calls
        if limit is None:
            return
        if tool_calls <= limit:
            return

        state.trace_event(
            component=TraceComponent.POLICY,
            step="budget_exceeded",
            level=TraceLevel.ERROR,
            message="Budget exceeded: max_tool_calls",
            payload=BudgetExceededDiagV1(
                run_id=run_id,
                budget_name="max_tool_calls",
                limit=limit,
                actual=tool_calls,
                enforcement_mode=self._policy.enforcement_mode.value,
            ),
        )

        if self._policy.enforcement_mode is BudgetEnforcementMode.ABORT:
            raise BudgetExceededError(f"Budget exceeded: max_tool_calls ({tool_calls} > {limit})")

