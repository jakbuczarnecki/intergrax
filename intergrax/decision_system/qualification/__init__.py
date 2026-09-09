# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Decision qualification failure taxonomy and reliability contracts (DS-E2E-14.3)."""

from intergrax.decision_system.qualification.axis_outcome import (
    DecisionQualificationAxisOutcome,
    derive_axis_outcomes,
)
from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassification,
    DecisionFailureClassificationAmbiguityError,
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.classifier import (
    CATEGORY_RULES,
    classify_decision_failure,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.reliability import (
    DecisionReliabilitySummary,
    aggregate_decision_reliability,
)
from intergrax.decision_system.qualification.run_result import (
    DecisionQualificationRunResult,
    DecisionQualificationRunResultConsistencyError,
    build_decision_qualification_run_result,
    category_from_run_result,
    validate_run_result_axis_consistency,
)
from intergrax.decision_system.qualification.serialization import (
    classification_to_dict,
    reliability_summary_to_dict,
    run_result_to_dict,
)
from intergrax.decision_system.qualification.signals import (
    EnvironmentQualificationSignal,
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
    ProviderQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import (
    BOUNDARY_ORDER,
    CATEGORY_PRECEDENCE,
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
    boundary_rank,
    earliest_boundary,
)

__all__ = [
    "BOUNDARY_ORDER",
    "CATEGORY_PRECEDENCE",
    "CATEGORY_RULES",
    "DecisionFailureBoundary",
    "DecisionFailureCategory",
    "DecisionFailureClassification",
    "DecisionFailureClassificationAmbiguityError",
    "DecisionFailureClassificationRule",
    "DecisionFailureDiagnosticCode",
    "DecisionFailureOwner",
    "DecisionFailureReason",
    "DecisionQualificationAxisOutcome",
    "DecisionQualificationObservation",
    "DecisionQualificationRunResult",
    "DecisionQualificationRunResultConsistencyError",
    "DecisionReliabilitySummary",
    "DecisionRetryability",
    "EnvironmentQualificationSignal",
    "EvaluatorQualificationSignal",
    "ModelBehaviorQualificationSignal",
    "ObservabilityQualificationSignal",
    "PlatformContractQualificationSignal",
    "ProviderQualificationSignal",
    "aggregate_decision_reliability",
    "boundary_rank",
    "build_decision_qualification_run_result",
    "category_from_run_result",
    "derive_axis_outcomes",
    "validate_run_result_axis_consistency",
    "classification_to_dict",
    "classify_decision_failure",
    "earliest_boundary",
    "reliability_summary_to_dict",
    "run_result_to_dict",
]
