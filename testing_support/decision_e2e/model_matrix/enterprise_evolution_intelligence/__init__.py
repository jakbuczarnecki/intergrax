# © Artur Czarnecki. All rights reserved.

"""Enterprise evolution intelligence — read-only analysis of adaptation history (L15)."""

from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.analyzer_providers import (
    AdaptationEffectivenessAnalyzer,
    CostImpactAnalyzer,
    GovernanceComplianceAnalyzer,
    StabilityAnalyzer,
    default_analyzer_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.audit_providers import (
    StandardEvolutionIntelligenceAuditProvider,
    default_evolution_intelligence_audit_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.contracts import (
    ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID,
    ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION,
    EvolutionAnalysisFinding,
    EvolutionGovernanceReference,
    EvolutionHistoryReference,
    EvolutionIntelligenceAuditMetadata,
    EvolutionIntelligenceContext,
    EvolutionIntelligenceDataSourceRef,
    EvolutionIntelligenceInsight,
    EvolutionIntelligenceRecommendation,
    EvolutionIntelligenceResult,
    EvolutionIntelligenceRunStatus,
    EvolutionMetricSnapshot,
    EvolutionRecommendationKind,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.engine import (
    EnterpriseEvolutionIntelligenceEngine,
    default_enterprise_evolution_intelligence_engine,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.insight_providers import (
    DefaultEvolutionInsightProvider,
    default_insight_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.intelligence_providers import (
    DefaultEnterpriseEvolutionIntelligenceProvider,
    default_enterprise_evolution_intelligence_provider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.metric_providers import (
    DefaultEvolutionMetricProvider,
    default_metric_providers,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.protocol import (
    EnterpriseEvolutionIntelligenceProvider,
    EvolutionAnalyzerProvider,
    EvolutionInsightProvider,
    EvolutionIntelligenceAuditProvider,
    EvolutionMetricProvider,
    EvolutionRecommendationProvider,
)
from testing_support.decision_e2e.model_matrix.enterprise_evolution_intelligence.recommendation_providers import (
    DefaultEvolutionRecommendationProvider,
    default_recommendation_providers,
)

__all__ = [
    "ENTERPRISE_EVOLUTION_INTELLIGENCE_TASK_ID",
    "ENTERPRISE_EVOLUTION_INTELLIGENCE_VERSION",
    "AdaptationEffectivenessAnalyzer",
    "CostImpactAnalyzer",
    "DefaultEnterpriseEvolutionIntelligenceProvider",
    "DefaultEvolutionInsightProvider",
    "DefaultEvolutionMetricProvider",
    "DefaultEvolutionRecommendationProvider",
    "EnterpriseEvolutionIntelligenceEngine",
    "EnterpriseEvolutionIntelligenceProvider",
    "EvolutionAnalysisFinding",
    "EvolutionAnalyzerProvider",
    "EvolutionGovernanceReference",
    "EvolutionHistoryReference",
    "EvolutionInsightProvider",
    "EvolutionIntelligenceAuditMetadata",
    "EvolutionIntelligenceAuditProvider",
    "EvolutionIntelligenceContext",
    "EvolutionIntelligenceDataSourceRef",
    "EvolutionIntelligenceInsight",
    "EvolutionIntelligenceRecommendation",
    "EvolutionIntelligenceResult",
    "EvolutionIntelligenceRunStatus",
    "EvolutionMetricProvider",
    "EvolutionMetricSnapshot",
    "EvolutionRecommendationKind",
    "EvolutionRecommendationProvider",
    "GovernanceComplianceAnalyzer",
    "StabilityAnalyzer",
    "StandardEvolutionIntelligenceAuditProvider",
    "default_analyzer_providers",
    "default_enterprise_evolution_intelligence_engine",
    "default_enterprise_evolution_intelligence_provider",
    "default_evolution_intelligence_audit_provider",
    "default_insight_providers",
    "default_metric_providers",
    "default_recommendation_providers",
]
