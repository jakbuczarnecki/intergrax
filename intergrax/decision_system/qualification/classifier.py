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
from intergrax.decision_system.qualification.taxonomy import (
    CATEGORY_PRECEDENCE,
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
    earliest_boundary,
)


def _build(
    *,
    category: DecisionFailureCategory,
    reason: DecisionFailureReason,
    boundary: DecisionFailureBoundary,
    owner: DecisionFailureOwner,
    retryability: DecisionRetryability,
    diagnostic_code: DecisionFailureDiagnosticCode,
) -> DecisionFailureClassification:
    return DecisionFailureClassification(
        category=category,
        reason=reason,
        boundary=boundary,
        owner=owner,
        retryability=retryability,
        diagnostic_code=diagnostic_code,
    )


def _resolve_boundary(
    observation: DecisionQualificationObservation,
    default: DecisionFailureBoundary,
    explicit: DecisionFailureBoundary | None,
) -> DecisionFailureBoundary:
    if explicit is not None:
        return explicit
    return earliest_boundary(observation.boundary, default)


def _model_behavior_boundary(
    observation: DecisionQualificationObservation,
    default: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    return _resolve_boundary(
        observation,
        default,
        observation.model_behavior.behavior_boundary,
    )


def _platform_violation_boundary(
    observation: DecisionQualificationObservation,
    default: DecisionFailureBoundary,
) -> DecisionFailureBoundary:
    return _resolve_boundary(
        observation,
        default,
        observation.platform_contract.violation_boundary,
    )


@dataclass(frozen=True, slots=True)
class _EnvironmentCredentialUnavailableRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.environment.credential_unavailable:
            return None
        return _build(
            category=DecisionFailureCategory.ENVIRONMENT,
            reason=DecisionFailureReason.CREDENTIAL_UNAVAILABLE,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
                None,
            ),
            owner=DecisionFailureOwner.ENVIRONMENT,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_CREDENTIAL_UNAVAILABLE,
        )


@dataclass(frozen=True, slots=True)
class _EnvironmentProviderConfigurationRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.environment.provider_configuration_invalid:
            return None
        return _build(
            category=DecisionFailureCategory.ENVIRONMENT,
            reason=DecisionFailureReason.PROVIDER_CONFIGURATION_INVALID,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.PROVIDER_BINDING,
                None,
            ),
            owner=DecisionFailureOwner.ENVIRONMENT,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_PROVIDER_CONFIGURATION,
        )


@dataclass(frozen=True, slots=True)
class _EnvironmentModelConfigurationRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.environment.model_configuration_invalid:
            return None
        return _build(
            category=DecisionFailureCategory.ENVIRONMENT,
            reason=DecisionFailureReason.MODEL_CONFIGURATION_INVALID,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.PROVIDER_BINDING,
                None,
            ),
            owner=DecisionFailureOwner.ENVIRONMENT,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_MODEL_CONFIGURATION,
        )


@dataclass(frozen=True, slots=True)
class _EnvironmentQualificationDisabledRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.environment.qualification_disabled:
            return None
        return _build(
            category=DecisionFailureCategory.ENVIRONMENT,
            reason=DecisionFailureReason.QUALIFICATION_DISABLED,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
                None,
            ),
            owner=DecisionFailureOwner.ENVIRONMENT,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.ENVIRONMENT_QUALIFICATION_DISABLED,
        )


@dataclass(frozen=True, slots=True)
class _ProviderRateLimitRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.provider.rate_limit:
            return None
        return _build(
            category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            reason=DecisionFailureReason.RATE_LIMIT,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.PROVIDER_BINDING,
                None,
            ),
            owner=DecisionFailureOwner.PROVIDER,
            retryability=DecisionRetryability.RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_RATE_LIMIT,
        )


@dataclass(frozen=True, slots=True)
class _ProviderTimeoutRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.provider.timeout:
            return None
        return _build(
            category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            reason=DecisionFailureReason.TIMEOUT,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.HOST_EXECUTION,
                None,
            ),
            owner=DecisionFailureOwner.PROVIDER,
            retryability=DecisionRetryability.RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_TIMEOUT,
        )


@dataclass(frozen=True, slots=True)
class _ProviderNetworkFailureRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.provider.network_failure:
            return None
        return _build(
            category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            reason=DecisionFailureReason.NETWORK_FAILURE,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.PROVIDER_BINDING,
                None,
            ),
            owner=DecisionFailureOwner.PROVIDER,
            retryability=DecisionRetryability.RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_NETWORK_FAILURE,
        )


@dataclass(frozen=True, slots=True)
class _ProviderServerErrorRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.provider.server_error:
            return None
        return _build(
            category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            reason=DecisionFailureReason.PROVIDER_SERVER_ERROR,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.HOST_EXECUTION,
                None,
            ),
            owner=DecisionFailureOwner.PROVIDER,
            retryability=DecisionRetryability.RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_SERVER_ERROR,
        )


@dataclass(frozen=True, slots=True)
class _ProviderProtocolErrorRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.provider.protocol_error:
            return None
        return _build(
            category=DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            reason=DecisionFailureReason.PROVIDER_PROTOCOL_ERROR,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.PROVIDER_BINDING,
                None,
            ),
            owner=DecisionFailureOwner.PROVIDER,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PROVIDER_PROTOCOL_ERROR,
        )


@dataclass(frozen=True, slots=True)
class _PlatformTraceNotFinalizedRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if observation.platform_contract.trace_finalized:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.TRACE_RUN_NOT_FINALIZED,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.TRACE_FINALIZATION,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TRACE_NOT_FINALIZED,
        )


@dataclass(frozen=True, slots=True)
class _PlatformWrongExecutionRouteRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.wrong_execution_route:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.WRONG_EXECUTION_ROUTE,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
            ),
            owner=DecisionFailureOwner.DECISION_SYSTEM,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_WRONG_EXECUTION_ROUTE,
        )


@dataclass(frozen=True, slots=True)
class _PlatformStrictToolContractRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.strict_tool_contract_violation:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.STRICT_TOOL_CONTRACT_VIOLATION,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_STRICT_TOOL_VIOLATION,
        )


@dataclass(frozen=True, slots=True)
class _PlatformToolDispatchContractRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.tool_dispatch_contract_violation:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.TOOL_DISPATCH_CONTRACT_VIOLATION,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.TOOL_DISPATCH,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TOOL_DISPATCH_VIOLATION,
        )


@dataclass(frozen=True, slots=True)
class _PlatformInvalidPhaseTransitionRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.invalid_phase_transition:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.INVALID_PHASE_TRANSITION,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.PHASE_VALIDATION,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_INVALID_PHASE_TRANSITION,
        )


@dataclass(frozen=True, slots=True)
class _PlatformCompletionReconciliationRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.completion_reconciliation_contract_violation:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.COMPLETION_RECONCILIATION_CONTRACT_VIOLATION,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.COMPLETION_RECONCILIATION,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_COMPLETION_RECONCILIATION,
        )


@dataclass(frozen=True, slots=True)
class _PlatformTerminalAcceptanceRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.platform_contract.terminal_acceptance_contract_violation:
            return None
        return _build(
            category=DecisionFailureCategory.PLATFORM_CONTRACT,
            reason=DecisionFailureReason.TERMINAL_ACCEPTANCE_CONTRACT_VIOLATION,
            boundary=_platform_violation_boundary(
                observation,
                DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
            ),
            owner=DecisionFailureOwner.EXECUTION_ENGINE,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.PLATFORM_TERMINAL_ACCEPTANCE,
        )


@dataclass(frozen=True, slots=True)
class _ModelInsufficientEvidenceRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.model_behavior.insufficient_evidence_gathering:
            return None
        return _build(
            category=DecisionFailureCategory.MODEL_BEHAVIOR,
            reason=DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING,
            boundary=_model_behavior_boundary(
                observation,
                DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
            ),
            owner=DecisionFailureOwner.MODEL,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.MODEL_INSUFFICIENT_EVIDENCE,
        )


@dataclass(frozen=True, slots=True)
class _ModelToolUseDeficiencyRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.model_behavior.tool_use_deficiency:
            return None
        return _build(
            category=DecisionFailureCategory.MODEL_BEHAVIOR,
            reason=DecisionFailureReason.TOOL_USE_DEFICIENCY,
            boundary=_model_behavior_boundary(
                observation,
                DecisionFailureBoundary.TOOL_DISPATCH,
            ),
            owner=DecisionFailureOwner.MODEL,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.MODEL_TOOL_USE_DEFICIENCY,
        )


@dataclass(frozen=True, slots=True)
class _ModelEpistemicContradictionRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.model_behavior.epistemic_contradiction:
            return None
        return _build(
            category=DecisionFailureCategory.MODEL_BEHAVIOR,
            reason=DecisionFailureReason.EPISTEMIC_CONTRADICTION,
            boundary=_model_behavior_boundary(
                observation,
                DecisionFailureBoundary.COMPLETION_RECONCILIATION,
            ),
            owner=DecisionFailureOwner.MODEL,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.MODEL_EPISTEMIC_CONTRADICTION,
        )


@dataclass(frozen=True, slots=True)
class _ModelUnsupportedCompletionRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.model_behavior.unsupported_completion:
            return None
        return _build(
            category=DecisionFailureCategory.MODEL_BEHAVIOR,
            reason=DecisionFailureReason.UNSUPPORTED_COMPLETION,
            boundary=_model_behavior_boundary(
                observation,
                DecisionFailureBoundary.REASONING,
            ),
            owner=DecisionFailureOwner.MODEL,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.MODEL_UNSUPPORTED_COMPLETION,
        )


@dataclass(frozen=True, slots=True)
class _ModelPrematureCompletionRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.model_behavior.premature_completion:
            return None
        return _build(
            category=DecisionFailureCategory.MODEL_BEHAVIOR,
            reason=DecisionFailureReason.PREMATURE_COMPLETION,
            boundary=_model_behavior_boundary(
                observation,
                DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
            ),
            owner=DecisionFailureOwner.MODEL,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.MODEL_PREMATURE_COMPLETION,
        )


@dataclass(frozen=True, slots=True)
class _EvaluatorFalseNegativeRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.evaluator.false_negative:
            return None
        return _build(
            category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
            reason=DecisionFailureReason.FALSE_NEGATIVE,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.EVALUATOR,
                None,
            ),
            owner=DecisionFailureOwner.EVALUATOR,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_FALSE_NEGATIVE,
        )


@dataclass(frozen=True, slots=True)
class _EvaluatorFalsePositiveRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.evaluator.false_positive:
            return None
        return _build(
            category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
            reason=DecisionFailureReason.FALSE_POSITIVE,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.EVALUATOR,
                None,
            ),
            owner=DecisionFailureOwner.EVALUATOR,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_FALSE_POSITIVE,
        )


@dataclass(frozen=True, slots=True)
class _EvaluatorCriterionSemanticsRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.evaluator.criterion_semantics_invalid:
            return None
        return _build(
            category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
            reason=DecisionFailureReason.CRITERION_SEMANTICS_INVALID,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.EVALUATOR,
                None,
            ),
            owner=DecisionFailureOwner.EVALUATOR,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_CRITERION_SEMANTICS,
        )


@dataclass(frozen=True, slots=True)
class _EvaluatorContractErrorRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.evaluator.contract_error:
            return None
        return _build(
            category=DecisionFailureCategory.EVALUATOR_SEMANTICS,
            reason=DecisionFailureReason.EVALUATOR_CONTRACT_ERROR,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.EVALUATOR,
                None,
            ),
            owner=DecisionFailureOwner.EVALUATOR,
            retryability=DecisionRetryability.NON_RETRIABLE,
            diagnostic_code=DecisionFailureDiagnosticCode.EVALUATOR_CONTRACT_ERROR,
        )


@dataclass(frozen=True, slots=True)
class _ObservabilityMissingSignalRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.observability.missing_required_signal:
            return None
        return _build(
            category=DecisionFailureCategory.OBSERVABILITY_GAP,
            reason=DecisionFailureReason.MISSING_REQUIRED_SIGNAL,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.TRACE_FINALIZATION,
                None,
            ),
            owner=DecisionFailureOwner.OBSERVABILITY,
            retryability=DecisionRetryability.UNKNOWN,
            diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_MISSING_SIGNAL,
        )


@dataclass(frozen=True, slots=True)
class _ObservabilityIncompleteTraceRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not observation.observability.incomplete_trace:
            return None
        return _build(
            category=DecisionFailureCategory.OBSERVABILITY_GAP,
            reason=DecisionFailureReason.INCOMPLETE_TRACE,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.TRACE_FINALIZATION,
                None,
            ),
            owner=DecisionFailureOwner.OBSERVABILITY,
            retryability=DecisionRetryability.UNKNOWN,
            diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_INCOMPLETE_TRACE,
        )


@dataclass(frozen=True, slots=True)
class _ObservabilityAmbiguousBoundaryRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if not (
            observation.observability.ambiguous_failure_boundary
            or observation.observability.critical_boundary_unknown
        ):
            return None
        return _build(
            category=DecisionFailureCategory.OBSERVABILITY_GAP,
            reason=DecisionFailureReason.AMBIGUOUS_FAILURE_BOUNDARY,
            boundary=_resolve_boundary(
                observation,
                DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
                None,
            ),
            owner=DecisionFailureOwner.OBSERVABILITY,
            retryability=DecisionRetryability.UNKNOWN,
            diagnostic_code=DecisionFailureDiagnosticCode.OBSERVABILITY_AMBIGUOUS_BOUNDARY,
        )


@dataclass(frozen=True, slots=True)
class _UnclassifiedFallbackRule:
    def classify(
        self,
        observation: DecisionQualificationObservation,
    ) -> DecisionFailureClassification | None:
        if observation.evaluator.passed is not False:
            return None
        if _has_explicit_failure_signal(observation):
            return None
        return _build(
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
    DecisionFailureCategory.ENVIRONMENT: (
        _EnvironmentCredentialUnavailableRule(),
        _EnvironmentProviderConfigurationRule(),
        _EnvironmentModelConfigurationRule(),
        _EnvironmentQualificationDisabledRule(),
    ),
    DecisionFailureCategory.PROVIDER_INFRASTRUCTURE: (
        _ProviderRateLimitRule(),
        _ProviderTimeoutRule(),
        _ProviderNetworkFailureRule(),
        _ProviderServerErrorRule(),
        _ProviderProtocolErrorRule(),
    ),
    DecisionFailureCategory.PLATFORM_CONTRACT: (
        _PlatformTraceNotFinalizedRule(),
        _PlatformWrongExecutionRouteRule(),
        _PlatformStrictToolContractRule(),
        _PlatformToolDispatchContractRule(),
        _PlatformInvalidPhaseTransitionRule(),
        _PlatformCompletionReconciliationRule(),
        _PlatformTerminalAcceptanceRule(),
    ),
    DecisionFailureCategory.MODEL_BEHAVIOR: (
        _ModelInsufficientEvidenceRule(),
        _ModelToolUseDeficiencyRule(),
        _ModelEpistemicContradictionRule(),
        _ModelUnsupportedCompletionRule(),
        _ModelPrematureCompletionRule(),
    ),
    DecisionFailureCategory.EVALUATOR_SEMANTICS: (
        _EvaluatorFalseNegativeRule(),
        _EvaluatorFalsePositiveRule(),
        _EvaluatorCriterionSemanticsRule(),
        _EvaluatorContractErrorRule(),
    ),
    DecisionFailureCategory.OBSERVABILITY_GAP: (
        _ObservabilityMissingSignalRule(),
        _ObservabilityIncompleteTraceRule(),
        _ObservabilityAmbiguousBoundaryRule(),
    ),
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
