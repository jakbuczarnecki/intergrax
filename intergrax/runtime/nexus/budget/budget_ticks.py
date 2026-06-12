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

from intergrax.runtime.nexus.budget.budget_enforcer import BudgetEnforcer
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState


def _enforcer(state: RuntimeState) -> BudgetEnforcer | None:
    cfg = state.context.config
    if cfg.run_budget is None or cfg.budget_policy is None:
        return None
    return BudgetEnforcer(cfg.run_budget, cfg.budget_policy)


def record_rag_invocation_and_enforce(state: RuntimeState) -> None:
    """After confirming RAG is enabled and configured; before retrieval work."""
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
    state.websearch_step_invocation_count += 1
    enc = _enforcer(state)
    if enc is not None:
        enc.check_websearch_invocations(
            run_id=state.run_id,
            websearch_invocations=state.websearch_step_invocation_count,
            state=state,
        )


def enforce_tool_call_budget(state: RuntimeState) -> None:
    """After each tool trace is appended (same metric as end-of-run ``check_tool_calls``)."""
    enc = _enforcer(state)
    if enc is not None:
        enc.check_tool_calls(
            run_id=state.run_id,
            tool_calls=len(state.tool_traces),
            state=state,
        )
