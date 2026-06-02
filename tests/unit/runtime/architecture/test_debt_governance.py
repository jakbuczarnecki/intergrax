from __future__ import annotations

from intergrax.runtime.architecture.architecture_metrics import compute_architecture_metrics
from intergrax.runtime.architecture.capability_graph import CapabilityGraph
from intergrax.runtime.architecture.debt_governance import (
    ArchitectureDebtReviewPolicy,
    DebtReviewCadence,
    evaluate_architecture_debt_governance,
)


def test_debt_governance_fails_without_runbook() -> None:
    report = compute_architecture_metrics(CapabilityGraph(nodes=[], edges=[]))
    governance = evaluate_architecture_debt_governance(
        metrics_report=report,
        policy=ArchitectureDebtReviewPolicy(
            cadence=DebtReviewCadence.BIWEEKLY,
            max_debt_index=0.50,
            owner_team="harness-architecture",
            runbook_ref="",
        ),
    )
    assert governance.result.passed is False
    assert any("runbook" in reason.lower() for reason in governance.result.reasons)


def test_debt_governance_passes_with_valid_policy_and_low_debt() -> None:
    report = compute_architecture_metrics(CapabilityGraph(nodes=[], edges=[]))
    report.summary.architecture_debt_index = 0.20
    governance = evaluate_architecture_debt_governance(
        metrics_report=report,
        policy=ArchitectureDebtReviewPolicy(
            cadence=DebtReviewCadence.WEEKLY,
            max_debt_index=0.50,
            owner_team="harness-architecture",
            runbook_ref="runbook/architecture/debt_review.md",
        ),
    )
    assert governance.result.passed is True
