# © Artur Czarnecki. All rights reserved.

"""Architecture debt governance and periodic review contracts (Phase V-AM.4)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.architecture_metrics import ArchitectureMetricsReport


class DebtReviewCadence(str, Enum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    MONTHLY = "monthly"


class ArchitectureDebtReviewPolicy(BaseModel):
    cadence: DebtReviewCadence = DebtReviewCadence.BIWEEKLY
    max_debt_index: float = 0.50
    owner_team: str
    runbook_ref: str


class ArchitectureDebtReviewResult(BaseModel):
    review_due_at: datetime
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class ArchitectureDebtGovernanceReport(BaseModel):
    schema_version: str = "1.0.0"
    policy: ArchitectureDebtReviewPolicy
    current_debt_index: float
    result: ArchitectureDebtReviewResult


def evaluate_architecture_debt_governance(
    *,
    metrics_report: ArchitectureMetricsReport,
    policy: ArchitectureDebtReviewPolicy,
    reviewed_at: datetime | None = None,
) -> ArchitectureDebtGovernanceReport:
    review_time = reviewed_at or datetime.now(UTC)
    due_at = _next_due_at(review_time, policy.cadence)
    debt_index = metrics_report.summary.architecture_debt_index
    reasons: list[str] = []
    if debt_index > policy.max_debt_index:
        reasons.append(
            "Architecture debt index above policy threshold: "
            f"{debt_index:.3f} > {policy.max_debt_index:.3f}"
        )
    if not policy.runbook_ref:
        reasons.append("Debt governance policy requires runbook reference")
    result = ArchitectureDebtReviewResult(
        review_due_at=due_at,
        passed=not reasons,
        reasons=reasons,
    )
    return ArchitectureDebtGovernanceReport(
        policy=policy,
        current_debt_index=debt_index,
        result=result,
    )


def _next_due_at(review_time: datetime, cadence: DebtReviewCadence) -> datetime:
    if cadence == DebtReviewCadence.WEEKLY:
        return review_time + timedelta(days=7)
    if cadence == DebtReviewCadence.BIWEEKLY:
        return review_time + timedelta(days=14)
    return review_time + timedelta(days=30)
