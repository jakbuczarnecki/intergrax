# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Predictive context intelligence contracts (PREDICTIVE R4)."""

from intergrax.contracts.predictive.analyzer_quality_profile import (
    PredictiveAnalyzerQualityProfile,
    default_analyzer_quality_profile,
)
from intergrax.contracts.predictive.audit import PredictionAuditRecord
from intergrax.contracts.predictive.completeness import PredictiveContextCompleteness
from intergrax.contracts.predictive.context_quality import PredictiveContextQualityReport
from intergrax.contracts.predictive.outcome_evaluation import (
    PredictionOutcomeEvaluation,
    PredictionOutcomeEvaluationResult,
)
from intergrax.contracts.predictive.plugin_registration import PredictivePluginRegistration
from intergrax.contracts.predictive.provenance import PredictiveContextProvenance
from intergrax.contracts.predictive.quality import (
    PredictiveQualityAssessment,
    PredictiveQualityDimension,
)
from intergrax.contracts.predictive.snapshot import PredictiveContextSnapshot

from intergrax.contracts.predictive.context import (
    PREDICTIVE_CONTEXT_VERSION,
    PredictiveContext,
    PredictiveContextMetadata,
    predictive_context_from_legacy_fields,
)
from intergrax.contracts.predictive.provider import (
    PredictiveContextProvider,
    PredictiveContextProviderFragment,
    PredictiveContextProviderStatus,
)
from intergrax.contracts.predictive.scope import PredictiveScope
from intergrax.contracts.predictive.sections import (
    PredictiveContextDiagnostic,
    PredictiveContextHistory,
    PredictiveContextPerformance,
)
from intergrax.contracts.predictive.types import (
    ExecutionPatternSnapshot,
    HistoricalProblemRef,
    PerformanceMetricPoint,
    PredictiveFindingRef,
    PredictiveRiskSignalRef,
)

__all__ = [
    "PREDICTIVE_CONTEXT_VERSION",
    "PredictionAuditRecord",
    "PredictionOutcomeEvaluation",
    "PredictionOutcomeEvaluationResult",
    "PredictiveAnalyzerQualityProfile",
    "PredictiveContextProvenance",
    "PredictiveContextQualityReport",
    "PredictiveContextSnapshot",
    "PredictivePluginRegistration",
    "PredictiveQualityAssessment",
    "PredictiveQualityDimension",
    "default_analyzer_quality_profile",
    "ExecutionPatternSnapshot",
    "HistoricalProblemRef",
    "PerformanceMetricPoint",
    "PredictiveContext",
    "PredictiveContextCompleteness",
    "PredictiveContextDiagnostic",
    "PredictiveContextHistory",
    "PredictiveContextMetadata",
    "PredictiveContextPerformance",
    "PredictiveContextProvider",
    "PredictiveContextProviderFragment",
    "PredictiveContextProviderStatus",
    "PredictiveFindingRef",
    "PredictiveRiskSignalRef",
    "PredictiveScope",
    "predictive_context_from_legacy_fields",
]
