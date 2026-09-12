# © Artur Czarnecki. All rights reserved.

"""Decision optimization learning loop — suggestions only (DS-E2E-15J-L9)."""

from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.contracts import (
    OPTIMIZATION_TASK_ID,
    OPTIMIZATION_VERSION,
    ConfidenceLevel,
    DecisionOptimizationAuditMetadata,
    DecisionOptimizationContext,
    DecisionOptimizationResult,
    DecisionOptimizationSuggestion,
    DetectedOptimizationPattern,
    OptimizationArea,
    OptimizationDataSourceKind,
    OptimizationDataSourceRef,
    OptimizationInsight,
    OptimizationRunStatus,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.engine import (
    DecisionOptimizationEngine,
    default_decision_optimization_engine,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.insight_generators import (
    GovernanceFrictionInsightGenerator,
    PatternLinkageInsightGenerator,
    default_insight_generators,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.pattern_analyzers import (
    GovernanceFrictionPatternAnalyzer,
    ModelCapabilityPatternAnalyzer,
    OutcomeReliabilityPatternAnalyzer,
    default_pattern_analyzers,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.protocol import (
    OptimizationInsightGenerator,
    OptimizationPatternAnalyzer,
    OptimizationRecommendationProvider,
)
from testing_support.decision_e2e.model_matrix.decision_optimization_learning_loop.recommendation_providers import (
    HumanReviewRecommendationProvider,
    default_recommendation_providers,
)

__all__ = [
    "OPTIMIZATION_TASK_ID",
    "OPTIMIZATION_VERSION",
    "ConfidenceLevel",
    "DecisionOptimizationAuditMetadata",
    "DecisionOptimizationContext",
    "DecisionOptimizationEngine",
    "DecisionOptimizationResult",
    "DecisionOptimizationSuggestion",
    "DetectedOptimizationPattern",
    "GovernanceFrictionInsightGenerator",
    "GovernanceFrictionPatternAnalyzer",
    "HumanReviewRecommendationProvider",
    "ModelCapabilityPatternAnalyzer",
    "OptimizationArea",
    "OptimizationDataSourceKind",
    "OptimizationDataSourceRef",
    "OptimizationInsight",
    "OptimizationInsightGenerator",
    "OptimizationPatternAnalyzer",
    "OptimizationRecommendationProvider",
    "OptimizationRunStatus",
    "OutcomeReliabilityPatternAnalyzer",
    "PatternLinkageInsightGenerator",
    "default_decision_optimization_engine",
    "default_insight_generators",
    "default_pattern_analyzers",
    "default_recommendation_providers",
]
