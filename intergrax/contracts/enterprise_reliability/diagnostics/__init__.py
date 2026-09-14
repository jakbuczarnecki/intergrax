# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""
Public ERL reliability operator diagnostics contracts (ERL-DIAG-001A).

Package placement (OQ-1 DECIDED): ``intergrax.contracts.enterprise_reliability.diagnostics``
— ERL is the source of diagnostic facts; central diagnostics consumes these contracts
without creating a dependency cycle with ``intergrax.contracts.diagnostics`` persistence ports.
"""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.correlation import (
    ReliabilityDiagnosticCorrelation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.emitter import (
    ExternalEffectReliabilityDiagnosticEmitter,
    NullExternalEffectReliabilityDiagnosticEmitter,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1,
    ExternalEffectReliabilityObservation,
    ExternalEffectReliabilityObservationValidationError,
    MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS,
)
from intergrax.contracts.enterprise_reliability.diagnostics.grouping import (
    ExternalEffectReliabilityProblemGroupingStrategy,
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID,
    RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION,
    ReliabilityCaseSubjectRef,
    ReliabilityProblemGroupingStrategyId,
    ReliabilityProblemGroupingStrategyVersion,
    parse_reliability_diagnostic_occurrence_instance_id,
    reliability_case_subject_index_token,
    reliability_correlation_subject_index_token,
    reliability_diagnostic_occurrence_instance_id,
)
from intergrax.contracts.enterprise_reliability.diagnostics.classification import (
    ExternalEffectReliabilityDiagnosticClassificationContext,
    ExternalEffectReliabilityDiagnosticClassificationResult,
    ExternalEffectReliabilityDiagnosticRecommendationStrategy,
    ExternalEffectReliabilityDiagnosticSeverity,
    ExternalEffectReliabilityDiagnosticSeverityStrategy,
    ExternalEffectReliabilityOperatorRecommendationKind,
    ExternalEffectReliabilityRecommendationDecision,
    ExternalEffectReliabilitySeverityDecision,
    MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN,
    RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID,
    RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION,
    RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID,
    RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION,
    ReliabilityDiagnosticRecommendationStrategyId,
    ReliabilityDiagnosticRecommendationStrategyVersion,
    ReliabilityDiagnosticSeverityStrategyId,
    ReliabilityDiagnosticSeverityStrategyVersion,
    build_reliability_diagnostic_classification_context,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    SCHEMA_AUTOMATION_SAFETY_HINT_V1,
    SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1,
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)

__all__ = [
    "AutomationSafetyHint",
    "ExternalEffectReliabilityDiagnosticClassificationContext",
    "ExternalEffectReliabilityDiagnosticClassificationResult",
    "ExternalEffectReliabilityDiagnosticEmitter",
    "ExternalEffectReliabilityDiagnosticRecommendationStrategy",
    "ExternalEffectReliabilityDiagnosticSeverity",
    "ExternalEffectReliabilityDiagnosticSeverityStrategy",
    "ExternalEffectReliabilityOperatorRecommendationKind",
    "ExternalEffectReliabilityRecommendationDecision",
    "ExternalEffectReliabilitySeverityDecision",
    "MAX_RELIABILITY_CLASSIFICATION_EXPLANATION_LEN",
    "RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_ID",
    "RELIABILITY_CONSERVATIVE_RECOMMENDATION_STRATEGY_VERSION",
    "RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_ID",
    "RELIABILITY_CONSERVATIVE_SEVERITY_STRATEGY_VERSION",
    "ReliabilityDiagnosticRecommendationStrategyId",
    "ReliabilityDiagnosticRecommendationStrategyVersion",
    "ReliabilityDiagnosticSeverityStrategyId",
    "ReliabilityDiagnosticSeverityStrategyVersion",
    "build_reliability_diagnostic_classification_context",
    "ExternalEffectReliabilityObservation",
    "ExternalEffectReliabilityObservationValidationError",
    "ExternalEffectReliabilityProblemGroupingStrategy",
    "ExternalEffectReliabilitySignalKind",
    "MAX_RELIABILITY_DIAGNOSTIC_TRACE_REFS",
    "NullExternalEffectReliabilityDiagnosticEmitter",
    "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_ID",
    "RELIABILITY_CASE_DEFAULT_GROUPING_STRATEGY_VERSION",
    "ReliabilityCaseSubjectRef",
    "ReliabilityDiagnosticArtifactRefs",
    "ReliabilityDiagnosticCorrelation",
    "ReliabilityProblemGroupingStrategyId",
    "ReliabilityProblemGroupingStrategyVersion",
    "SCHEMA_AUTOMATION_SAFETY_HINT_V1",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_OBSERVATION_V1",
    "SCHEMA_EXTERNAL_EFFECT_RELIABILITY_SIGNAL_KIND_V1",
    "parse_reliability_diagnostic_occurrence_instance_id",
    "reliability_case_subject_index_token",
    "reliability_correlation_subject_index_token",
    "reliability_diagnostic_occurrence_instance_id",
]
