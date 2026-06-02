# © Artur Czarnecki. All rights reserved.

"""Budget envelope contracts for multi-scope cost governance (Phase V-COST.1)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BudgetScope(str, Enum):
    TENANT = "tenant"
    APPLICATION = "application"
    AGENT = "agent"
    MODEL = "model"
    TOOL = "tool"


class BudgetEnvelope(BaseModel):
    scope: BudgetScope
    scope_id: str
    currency: str = "USD"
    limit_amount: float
    spent_amount: float = 0.0


class BudgetEnvelopeDecision(BaseModel):
    scope: BudgetScope
    scope_id: str
    within_budget: bool
    remaining_amount: float
    reasons: list[str] = Field(default_factory=list)


class BudgetGovernanceReport(BaseModel):
    schema_version: str = "1.0.0"
    envelopes: list[BudgetEnvelope] = Field(default_factory=list)
    decisions: list[BudgetEnvelopeDecision] = Field(default_factory=list)


def evaluate_budget_envelopes(envelopes: list[BudgetEnvelope]) -> BudgetGovernanceReport:
    decisions: list[BudgetEnvelopeDecision] = []
    for envelope in envelopes:
        remaining = envelope.limit_amount - envelope.spent_amount
        within_budget = remaining >= 0.0
        reasons: list[str] = []
        if not within_budget:
            reasons.append(
                f"Budget exceeded for {envelope.scope.value}:{envelope.scope_id} "
                f"({envelope.spent_amount:.2f} > {envelope.limit_amount:.2f})"
            )
        decisions.append(
            BudgetEnvelopeDecision(
                scope=envelope.scope,
                scope_id=envelope.scope_id,
                within_budget=within_budget,
                remaining_amount=max(0.0, remaining),
                reasons=reasons,
            )
        )
    return BudgetGovernanceReport(envelopes=envelopes, decisions=decisions)
