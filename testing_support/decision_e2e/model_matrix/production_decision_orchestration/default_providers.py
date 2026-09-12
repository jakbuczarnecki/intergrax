# © Artur Czarnecki. All rights reserved.

"""Default engine-backed orchestration providers (DS-E2E-15J-L6)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.contracts import (
    GovernanceDecision,
    GovernanceEvaluationRequest,
)
from testing_support.decision_e2e.model_matrix.governance_controlled_model_routing.engine import (
    GovernanceEvaluationEngine,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    ModelSelectionRecommendation,
    ModelSelectionRequest,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.engine import (
    ModelSelectionEngine,
)
from testing_support.decision_e2e.model_matrix.production_decision_orchestration.contracts import (
    DecisionExecutionRequest,
    DecisionExecutionResultReference,
    DecisionExecutionStatus,
)


@dataclass(frozen=True, slots=True)
class EngineBackedSelectionProvider:
    """``DecisionSelectionProvider`` delegating to ``ModelSelectionEngine``."""

    engine: ModelSelectionEngine

    @property
    def provider_id(self) -> str:
        return "engine-backed-selection"

    def recommend(
        self,
        request: ModelSelectionRequest,
        *,
        recommended_at: datetime | None = None,
    ) -> ModelSelectionRecommendation:
        return self.engine.recommend(request, recommended_at=recommended_at)


@dataclass(frozen=True, slots=True)
class EngineBackedGovernanceProvider:
    """``GovernanceDecisionProvider`` delegating to ``GovernanceEvaluationEngine``."""

    engine: GovernanceEvaluationEngine

    @property
    def provider_id(self) -> str:
        return "engine-backed-governance"

    def evaluate(
        self,
        request: GovernanceEvaluationRequest,
        *,
        evaluated_at: datetime | None = None,
    ) -> GovernanceDecision:
        return self.engine.evaluate(request, evaluated_at=evaluated_at)


@dataclass(frozen=True, slots=True)
class RecordingExecutionProvider:
    """Minimal execution provider that records an auditable reference only."""

    @property
    def provider_id(self) -> str:
        return "recording-execution"

    def execute(
        self,
        request: DecisionExecutionRequest,
        *,
        executed_at: datetime | None = None,
    ) -> DecisionExecutionResultReference:
        profile_key = None
        selected = request.selection_result.selected_model_reference
        if selected is not None:
            profile_key = selected.profile_key
        return DecisionExecutionResultReference(
            execution_reference_id=uuid4().hex,
            provider_id=self.provider_id,
            status=DecisionExecutionStatus.EXECUTED,
            profile_key=profile_key,
            governance_decision_id=request.governance_decision.audit_metadata.decision_id,
            selection_task_id=request.selection_result.decision_metadata.selection_task_id,
        )


__all__ = [
    "EngineBackedGovernanceProvider",
    "EngineBackedSelectionProvider",
    "RecordingExecutionProvider",
]
