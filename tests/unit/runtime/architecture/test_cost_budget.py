from __future__ import annotations

from intergrax.runtime.architecture.cost_budget import (
    BudgetEnvelope,
    BudgetScope,
    evaluate_budget_envelopes,
)


def test_budget_envelope_detects_exceeded_scope() -> None:
    report = evaluate_budget_envelopes(
        [
            BudgetEnvelope(
                scope=BudgetScope.AGENT,
                scope_id="agent:research",
                limit_amount=100.0,
                spent_amount=120.0,
            )
        ]
    )
    assert report.decisions[0].within_budget is False
    assert report.decisions[0].reasons


def test_budget_envelope_passes_within_limit() -> None:
    report = evaluate_budget_envelopes(
        [
            BudgetEnvelope(
                scope=BudgetScope.TENANT,
                scope_id="tenant-a",
                limit_amount=1000.0,
                spent_amount=500.0,
            )
        ]
    )
    assert report.decisions[0].within_budget is True
    assert report.decisions[0].remaining_amount == 500.0
