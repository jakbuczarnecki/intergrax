# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Observability gap failure rule declarations (DS-E2E-14.3A)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules.contracts import (
    DecisionFailureRuleSpec,
    resolve_default_boundary,
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


def _missing_required_signal(observation: DecisionQualificationObservation) -> bool:
    return observation.observability.missing_required_signal


def _incomplete_trace(observation: DecisionQualificationObservation) -> bool:
    return observation.observability.incomplete_trace


def _ambiguous_failure_boundary(observation: DecisionQualificationObservation) -> bool:
    return (
        observation.observability.ambiguous_failure_boundary
        or observation.observability.critical_boundary_unknown
    )


_OBSERVABILITY_RULE_SPECS: tuple[DecisionFailureRuleSpec, ...] = (
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.OBSERVABILITY_GAP,
        reason=DecisionFailureReason.MISSING_REQUIRED_SIGNAL,
        owner=DecisionFailureOwner.OBSERVABILITY,
        retryability=DecisionRetryability.UNKNOWN,
        diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_MISSING_SIGNAL,
        default_boundary=DecisionFailureBoundary.TRACE_FINALIZATION,
        predicate=_missing_required_signal,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.OBSERVABILITY_GAP,
        reason=DecisionFailureReason.INCOMPLETE_TRACE,
        owner=DecisionFailureOwner.OBSERVABILITY,
        retryability=DecisionRetryability.UNKNOWN,
        diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_INCOMPLETE_TRACE,
        default_boundary=DecisionFailureBoundary.TRACE_FINALIZATION,
        predicate=_incomplete_trace,
        boundary_resolver=resolve_default_boundary,
    ),
    DecisionFailureRuleSpec(
        category=DecisionFailureCategory.OBSERVABILITY_GAP,
        reason=DecisionFailureReason.AMBIGUOUS_FAILURE_BOUNDARY,
        owner=DecisionFailureOwner.OBSERVABILITY,
        retryability=DecisionRetryability.UNKNOWN,
        diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_AMBIGUOUS_BOUNDARY,
        default_boundary=DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
        predicate=_ambiguous_failure_boundary,
        boundary_resolver=resolve_default_boundary,
    ),
)

OBSERVABILITY_RULES: tuple[DecisionFailureClassificationRule, ...] = tuple(
    rule_from_spec(spec) for spec in _OBSERVABILITY_RULE_SPECS
)
