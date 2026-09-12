# © Artur Czarnecki. All rights reserved.

"""Adaptive decision intelligence — advisory insights only (DS-E2E-15J-L10)."""

from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.contracts import (
    ADAPTIVE_INTELLIGENCE_TASK_ID,
    ADAPTIVE_INTELLIGENCE_VERSION,
    AdaptiveDataSourceKind,
    AdaptiveDataSourceRef,
    AdaptiveDecisionContextReference,
    AdaptiveDecisionInsight,
    AdaptiveDecisionIntelligenceAuditMetadata,
    AdaptiveDecisionIntelligenceContext,
    AdaptiveDecisionIntelligenceInput,
    AdaptiveDecisionIntelligenceResult,
    AdaptiveDecisionRecommendation,
    AdaptiveIntelligenceRunStatus,
    AdaptiveReasoningInsight,
    ConfidenceLevel,
    HistoricalEvidence,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.context_providers import (
    AnalyticsHistoryContextProvider,
    CapabilityContextProvider,
    GovernanceContextProvider,
    LifecycleHistoryContextProvider,
    OptimizationContextProvider,
    default_context_providers,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.engine import (
    AdaptiveDecisionIntelligenceEngine,
    default_adaptive_decision_intelligence_engine,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.protocol import (
    AdaptiveDecisionContextProvider,
    AdaptiveReasoningProvider,
    AdaptiveRecommendationProvider,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.reasoning_providers import (
    PerformanceReasoningProvider,
    QualityReasoningProvider,
    RiskReasoningProvider,
    default_reasoning_providers,
)
from testing_support.decision_e2e.model_matrix.adaptive_decision_intelligence.recommendation_providers import (
    BusinessRecommendationProvider,
    TechnicalRecommendationProvider,
    default_recommendation_providers,
)

__all__ = [
    "ADAPTIVE_INTELLIGENCE_TASK_ID",
    "ADAPTIVE_INTELLIGENCE_VERSION",
    "AdaptiveDataSourceKind",
    "AdaptiveDataSourceRef",
    "AdaptiveDecisionContextProvider",
    "AdaptiveDecisionContextReference",
    "AdaptiveDecisionInsight",
    "AdaptiveDecisionIntelligenceAuditMetadata",
    "AdaptiveDecisionIntelligenceContext",
    "AdaptiveDecisionIntelligenceEngine",
    "AdaptiveDecisionIntelligenceInput",
    "AdaptiveDecisionIntelligenceResult",
    "AdaptiveDecisionRecommendation",
    "AdaptiveIntelligenceRunStatus",
    "AdaptiveReasoningInsight",
    "AdaptiveReasoningProvider",
    "AdaptiveRecommendationProvider",
    "AnalyticsHistoryContextProvider",
    "BusinessRecommendationProvider",
    "CapabilityContextProvider",
    "ConfidenceLevel",
    "GovernanceContextProvider",
    "HistoricalEvidence",
    "LifecycleHistoryContextProvider",
    "OptimizationContextProvider",
    "PerformanceReasoningProvider",
    "QualityReasoningProvider",
    "RiskReasoningProvider",
    "TechnicalRecommendationProvider",
    "default_adaptive_decision_intelligence_engine",
    "default_context_providers",
    "default_reasoning_providers",
    "default_recommendation_providers",
]
