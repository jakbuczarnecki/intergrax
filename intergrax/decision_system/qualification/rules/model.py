# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Model behavior failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules.contracts import (
    DecisionFailureRuleSpec,
    resolve_model_behavior_boundary,
    rule_from_spec,
)
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


def _insufficient_evidence_gathering(
    observation: DecisionQualificationObservation,
) -> bool:
    return observation.model_behavior.insufficient_evidence_gathering


def _tool_use_deficiency(observation: DecisionQualificationObservation) -> bool:
    return observation.model_behavior.tool_use_deficiency


def _epistemic_contradiction(observation: DecisionQualificationObservation) -> bool:
    return observation.model_behavior.epistemic_contradiction


def _unsupported_completion(observation: DecisionQualificationObservation) -> bool:
    return observation.model_behavior.unsupported_completion


def _premature_completion(observation: DecisionQualificationObservation) -> bool:
    return observation.model_behavior.premature_completion


_MODEL_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING,
        owner=DecisionFailureOwner.MODEL,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.MODEL_INSUFFICIENT_EVIDENCE,
        default_boundary=DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
        predicate=_insufficient_evidence_gathering,
        boundary_resolver=resolve_model_behavior_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.TOOL_USE_DEFICIENCY,
        owner=DecisionFailureOwner.MODEL,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.MODEL_TOOL_USE_DEFICIENCY,
        default_boundary=DecisionFailureBoundary.TOOL_DISPATCH,
        predicate=_tool_use_deficiency,
        boundary_resolver=resolve_model_behavior_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.EPISTEMIC_CONTRADICTION,
        owner=DecisionFailureOwner.MODEL,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.MODEL_EPISTEMIC_CONTRADICTION,
        default_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        predicate=_epistemic_contradiction,
        boundary_resolver=resolve_model_behavior_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.UNSUPPORTED_COMPLETION,
        owner=DecisionFailureOwner.MODEL,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.MODEL_UNSUPPORTED_COMPLETION,
        default_boundary=DecisionFailureBoundary.REASONING,
        predicate=_unsupported_completion,
        boundary_resolver=resolve_model_behavior_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.MODEL_BEHAVIOR,
        reason=DecisionFailureReason.PREMATURE_COMPLETION,
        owner=DecisionFailureOwner.MODEL,
        retryability=DecisionRetryability.NON_RETRIABLE,
        diagnostic_code=DecisionFailureDiagnosticCode.MODEL_PREMATURE_COMPLETION,
        default_boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
        predicate=_premature_completion,
        boundary_resolver=resolve_model_behavior_boundary,
    ),
)

MODEL_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _MODEL_RULE_SPECS
)
