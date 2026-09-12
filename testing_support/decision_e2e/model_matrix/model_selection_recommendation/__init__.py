# © Artur Czarnecki. All rights reserved.

"""Model selection recommendation from capability baseline (DS-E2E-15J-L4)."""

from testing_support.decision_e2e.model_matrix.model_selection_recommendation.contracts import (
    SELECTION_TASK_ID,
    SELECTION_VERSION,
    CapabilitySelectionConstraints,
    MatchedCapabilityEvidence,
    ModelEvidenceReference,
    ModelSelectionDecisionMetadata,
    ModelSelectionRecommendation,
    ModelSelectionRequest,
    ModelSelectionStatus,
    StrategyParticipationRecord,
    TaskCapabilityRequirement,
    TaskRequirements,
    UnmetRequirementEvidence,
    observation_level_rank,
    observation_meets_minimum,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.engine import (
    ModelSelectionEngine,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.protocol import (
    SelectionStrategy,
    SelectionStrategyResult,
    StrategyModelAssessment,
)
from testing_support.decision_e2e.model_matrix.model_selection_recommendation.strategies import (
    CapabilityMatchStrategy,
    CostPreferenceStrategy,
    SafetyLimitationStrategy,
    default_selection_strategies,
)

__all__ = [
    "CapabilityMatchStrategy",
    "CapabilitySelectionConstraints",
    "CostPreferenceStrategy",
    "MatchedCapabilityEvidence",
    "ModelEvidenceReference",
    "ModelSelectionDecisionMetadata",
    "ModelSelectionEngine",
    "ModelSelectionRecommendation",
    "ModelSelectionRequest",
    "ModelSelectionStatus",
    "SafetyLimitationStrategy",
    "SELECTION_TASK_ID",
    "SELECTION_VERSION",
    "SelectionStrategy",
    "SelectionStrategyResult",
    "StrategyModelAssessment",
    "StrategyParticipationRecord",
    "TaskCapabilityRequirement",
    "TaskRequirements",
    "UnmetRequirementEvidence",
    "default_selection_strategies",
    "observation_level_rank",
    "observation_meets_minimum",
]
