# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic Decision qualification failure classifier (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.classification import (
    DecisionFailureClassification,
    DecisionFailureClassificationAmbiguityError,
    DecisionFailureClassificationRule,
)
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.rules import (
    ENVIRONMENT_RULES,
    EVALUATOR_RULES,
    MODEL_RULES,
    OBSERVABILITY_RULES,
    PLATFORM_RULES,
    PROVIDER_RULES,
)
from intergrax.decision_system.qualification.rules.contracts import build_classification
from intergrax.decision_system.qualification.taxonomy import (
    CATEGORY_PRECEDENCE,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


@dataclass(frozen=True, slots=True)
class _UnclassifiedFallbackRule:
    """Specialized rule: evaluator fail without any explicit failure signal."""

    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if observation.evaluator.passed is not False:
            return None
        if _has_explicit_failure_signal(observation):
            return None
        return build_classification(
            category=DecisionFailureCategory.UNCLASSIFIED,
            reason=DecisionFailureReason.UNCLASSIFIED,
            boundary=observation.boundary,
            owner=DecisionFailureOwner.DECISION_SYSTEM,
            retryability=DecisionRetryability.UNKNOWN,
            diagnostic_code=DecisionFailureDiagnosticCode.UNCLASSIFIED,
        )


def _has_explicit_failure_signal(observation: DecisionQualificationObservation) -> bool:
    return any(
        (
            observation.environment.credential_unavailable,
            observation.environment.provider_configuration_invalid,
            observation.environment.model_configuration_invalid,
            observation.environment.qualification_disabled,
            observation.provider.rate_limit,
            observation.provider.timeout,
            observation.provider.network_failure,
            observation.provider.server_error,
            observation.provider.protocol_error,
            not observation.platform_contract.trace_finalized,
            observation.platform_contract.wrong_execution_route,
            observation.platform_contract.strict_tool_contract_violation,
            observation.platform_contract.tool_dispatch_contract_violation,
            observation.platform_contract.invalid_phase_transition,
            observation.platform_contract.completion_reconciliation_contract_violation,
            observation.platform_contract.terminal_acceptance_contract_violation,
            observation.model_behavior.insufficient_evidence_gathering,
            observation.model_behavior.tool_use_deficiency,
            observation.model_behavior.epistemic_contradiction,
            observation.model_behavior.unsupported_completion,
            observation.model_behavior.premature_completion,
            observation.evaluator.false_negative,
            observation.evaluator.false_positive,
            observation.evaluator.criterion_semantics_invalid,
            observation.evaluator.contract_error,
            observation.observability.missing_required_signal,
            observation.observability.incomplete_trace,
            observation.observability.ambiguous_failure_boundary,
            observation.observability.critical_boundary_unknown,
        )
    )


CATEGORY_RULES: dict[DecisionFailureCategory, tuple[DecisionFailureClassificationRule, ...]] = {
    DecisionFailureCategory.ENVIRONMENT: ENVIRONMENT_RULES,
    DecisionFailureCategory.PROVIDER_INFRASTRUCTURE: PROVIDER_RULES,
    DecisionFailureCategory.PLATFORM_CONTRACT: PLATFORM_RULES,
    DecisionFailureCategory.MODEL_BEHAVIOR: MODEL_RULES,
    DecisionFailureCategory.EVALUATOR_SEMANTICS: EVALUATOR_RULES,
    DecisionFailureCategory.OBSERVABILITY_GAP: OBSERVABILITY_RULES,
    DecisionFailureCategory.UNCLASSIFIED: (_UnclassifiedFallbackRule(),),
}


def _collect_category_candidates(
    category: DecisionFailureCategory,
    observation: DecisionQualificationObservation,
) -> tuple[DecisionFailureClassification, ...]:
    matches: list[DecisionFailureClassification] = []
    for rule in CATEGORY_RULES[category]:
        match = rule.classify(observation)
        if match is not None:
            matches.append(match)
    return tuple(matches)


def _resolve_candidates(
    candidates: tuple[DecisionFailureClassification, ...],
) -> DecisionFailureClassification:
    root_boundary = min(candidate.boundary for candidate in candidates)
    at_root = tuple(
        candidate for candidate in candidates if candidate.boundary is root_boundary
    )
    reasons = {candidate.reason for candidate in at_root}
    if len(reasons) > 1:
        reason_list = ", ".join(sorted(reason.value for reason in reasons))
        raise DecisionFailureClassificationAmbiguityError(
            "ambiguous root failure classification at "
            f"{root_boundary.value}: {reason_list}"
        )
    return at_root[0]


def classify_decision_failure(
    observation: DecisionQualificationObservation,
) -> DecisionFailureClassification | None:
    """Pure classifier: structured observation facts to typed failure or pass."""
    if not observation.observability_complete:
        observability_candidates = _collect_category_candidates(
            DecisionFailureCategory.OBSERVABILITY_GAP,
            observation,
        )
        if observability_candidates:
            return _resolve_candidates(observability_candidates)

    for category in CATEGORY_PRECEDENCE:
        candidates = _collect_category_candidates(category, observation)
        if candidates:
            return _resolve_candidates(candidates)
    return None
