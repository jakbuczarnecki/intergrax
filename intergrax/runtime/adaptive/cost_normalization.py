# © Artur Czarnecki. All rights reserved.

"""Cost normalization helpers for adaptive signals (Phase W-ADAPT-1.7)."""

from __future__ import annotations

from intergrax.runtime.nexus.budget.budget_models import RunBudget


def normalize_cost_against_budget(
    *,
    total_tokens: int | None,
    actual_cost: float | None,
    run_budget: RunBudget | None,
) -> float:
    """
    Return actual/budget ratio (1.0 = at budget).

    Prefers token budget when available; falls back to a neutral ratio when no limit exists.
    """
    if run_budget is not None and run_budget.max_total_tokens and total_tokens is not None:
        return total_tokens / run_budget.max_total_tokens
    if run_budget is not None and run_budget.max_llm_calls is not None:
        return 0.0
    if actual_cost is not None and actual_cost > 0.0:
        return min(1.0, actual_cost)
    return 0.0
