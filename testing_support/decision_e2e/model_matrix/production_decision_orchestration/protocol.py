# © Artur Czarnecki. All rights reserved.

"""Pluggable provider contracts for decision orchestration (DS-E2E-15J-L6)."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDecision,
    GovernanceEvaluationRequest,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    ModelSelectionRecommendation,
    ModelSelectionRequest,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.contracts import (
    DecisionExecutionRequest,
    DecisionExecutionResultReference,
)


class DecisionSelectionProvider(Protocol):
    """Pluggable selection stage; produces a model recommendation."""

    @property
    def provider_id(self) -> str: ...

    def recommend(
        self,
        request: ModelSelectionRequest,
        *,
        recommended_at: datetime | None = None,
    ) -> ModelSelectionRecommendation: ...


class GovernanceDecisionProvider(Protocol):
    """Pluggable governance stage; evaluates whether selection may proceed."""

    @property
    def provider_id(self) -> str: ...

    def evaluate(
        self,
        request: GovernanceEvaluationRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> GovernanceDecision: ...


class ExecutionProvider(Protocol):
    """Pluggable execution stage; runs only after governance allows."""

    @property
    def provider_id(self) -> str: ...

    def execute(
        self,
        request: DecisionExecutionRequest,
        *,
        executed_at: datetime | None = None,
    ) -> DecisionExecutionResultReference: ...


__all__ = [
    "DecisionSelectionProvider",
    "ExecutionProvider",
    "GovernanceDecisionProvider",
]
