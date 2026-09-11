# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Preventive recommendation contracts — decision support only (PREVENTIVE R6)."""

from intergrax.contracts.preventive.analyzer_descriptor import (
    PreventiveAnalyzerDescriptor,
    PreventiveResourceBudget,
)
from intergrax.contracts.preventive.audit import PreventiveAuditRecord, PreventiveRecommendationAuditRecord
from intergrax.contracts.preventive.conflict import (
    CONFLICTING_RECOMMENDATIONS,
    PreventiveRecommendationConflict,
)
from intergrax.contracts.preventive.lifecycle import (
    PreventiveLifecycleViolation,
    PreventiveRecommendationLifecycleState,
    assert_lifecycle_transition,
)
from intergrax.contracts.preventive.safety import PreventiveSafetyAssessment, preventive_safety_always_disabled
from intergrax.contracts.preventive.category import PreventiveRecommendationCategory
from intergrax.contracts.preventive.context import (
    DiagnosticEvidenceContext,
    HistoricalOutcome,
    PreventiveAnalysisInput,
)
from intergrax.contracts.preventive.evidence import RecommendationEvidenceReference
from intergrax.contracts.preventive.governance import PreventiveRecommendationGovernance
from intergrax.contracts.preventive.outcome_evaluation import (
    OperatorRecommendationDecision,
    RecommendationEffectiveness,
    RecommendationOutcomeEvaluation,
)
from intergrax.contracts.preventive.preventive_analyzer import PreventiveAnalyzer
from intergrax.contracts.preventive.recommendation import (
    PreventiveRecommendation,
    PreventiveRecommendationCandidate,
    mint_preventive_recommendation_id,
)

__all__ = [
    "CONFLICTING_RECOMMENDATIONS",
    "DiagnosticEvidenceContext",
    "HistoricalOutcome",
    "OperatorRecommendationDecision",
    "PreventiveAnalysisInput",
    "PreventiveAnalyzer",
    "PreventiveAnalyzerDescriptor",
    "PreventiveAuditRecord",
    "PreventiveLifecycleViolation",
    "PreventiveRecommendation",
    "PreventiveRecommendationAuditRecord",
    "PreventiveRecommendationCandidate",
    "PreventiveRecommendationCategory",
    "PreventiveRecommendationConflict",
    "PreventiveRecommendationGovernance",
    "PreventiveRecommendationLifecycleState",
    "PreventiveResourceBudget",
    "PreventiveSafetyAssessment",
    "RecommendationEffectiveness",
    "RecommendationEvidenceReference",
    "RecommendationOutcomeEvaluation",
    "assert_lifecycle_transition",
    "mint_preventive_recommendation_id",
    "preventive_safety_always_disabled",
]
