# © Artur Czarnecki. All rights reserved.

"""Golden classification matrix and parity tests (DS-E2E-14.3A)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from intergrax.decision_system.qualification.classifier import classify_decision_failure
from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.serialization import classification_to_dict
from intergrax.decision_system.qualification.signals import (
    EnvironmentQualificationSignal,
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
    ProviderQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import (
    DecisionFailureBoundary,
    DecisionFailureCategory,
    DecisionFailureDiagnosticCode,
    DecisionFailureOwner,
    DecisionFailureReason,
    DecisionRetryability,
)


def _base_observation(**overrides: object) -> DecisionQualificationObservation:
    defaults = {
        "boundary": DecisionFailureBoundary.HOST_EXECUTION,
        "platform_contract": PlatformContractQualificationSignal(),
        "model_behavior": ModelBehaviorQualificationSignal(),
        "evaluator": EvaluatorQualificationSignal(passed=False),
        "provider": ProviderQualificationSignal(),
        "environment": EnvironmentQualificationSignal(),
        "observability": ObservabilityQualificationSignal(),
        "observability_complete": True,
    }
    defaults.update(overrides)
    return DecisionQualificationObservation(**defaults)  # type: ignore[arg-type]


def _assert_classification(
    observation: DecisionQualificationObservation,
    *,
    category: DecisionFailureCategory,
    reason: DecisionFailureReason,
    owner: DecisionFailureOwner,
    boundary: DecisionFailureBoundary,
    retryability: DecisionRetryability,
    diagnostic_code: DecisionFailureDiagnosticCode,
) -> None:
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is category
    assert result.reason is reason
    assert result.owner is owner
    assert result.boundary is boundary
    assert result.retryability is retryability
    assert result.diagnostic_code is diagnostic_code


_GOLDEN_CASES: tuple[
    tuple[str, Callable[[], DecisionQualificationObservation], dict[str, object]],
    ...,
] = (
    (
        "credential_unavailable",
        lambda: _base_observation(
            environment=EnvironmentQualificationSignal(credential_unavailable=True),
        ),
        {
            "category": DecisionFailureCategory.ENVIRONMENT,
            "reason": DecisionFailureReason.CREDENTIAL_UNAVAILABLE,
            "owner": DecisionFailureOwner.ENVIRONMENT,
            "boundary": DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.ENVIRONMENT_CREDENTIAL_UNAVAILABLE,
        },
    ),
    (
        "provider_config_invalid",
        lambda: _base_observation(
            environment=EnvironmentQualificationSignal(
                provider_configuration_invalid=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.ENVIRONMENT,
            "reason": DecisionFailureReason.PROVIDER_CONFIGURATION_INVALID,
            "owner": DecisionFailureOwner.ENVIRONMENT,
            "boundary": DecisionFailureBoundary.PROVIDER_BINDING,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.ENVIRONMENT_PROVIDER_CONFIGURATION,
        },
    ),
    (
        "model_config_invalid",
        lambda: _base_observation(
            environment=EnvironmentQualificationSignal(
                model_configuration_invalid=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.ENVIRONMENT,
            "reason": DecisionFailureReason.MODEL_CONFIGURATION_INVALID,
            "owner": DecisionFailureOwner.ENVIRONMENT,
            "boundary": DecisionFailureBoundary.PROVIDER_BINDING,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.ENVIRONMENT_MODEL_CONFIGURATION,
        },
    ),
    (
        "qualification_disabled",
        lambda: _base_observation(
            environment=EnvironmentQualificationSignal(qualification_disabled=True),
        ),
        {
            "category": DecisionFailureCategory.ENVIRONMENT,
            "reason": DecisionFailureReason.QUALIFICATION_DISABLED,
            "owner": DecisionFailureOwner.ENVIRONMENT,
            "boundary": DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.ENVIRONMENT_QUALIFICATION_DISABLED,
        },
    ),
    (
        "rate_limit",
        lambda: _base_observation(provider=ProviderQualificationSignal(rate_limit=True)),
        {
            "category": DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            "reason": DecisionFailureReason.RATE_LIMIT,
            "owner": DecisionFailureOwner.PROVIDER,
            "boundary": DecisionFailureBoundary.PROVIDER_BINDING,
            "retryability": DecisionRetryability.RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PROVIDER_RATE_LIMIT,
        },
    ),
    (
        "timeout",
        lambda: _base_observation(provider=ProviderQualificationSignal(timeout=True)),
        {
            "category": DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            "reason": DecisionFailureReason.TIMEOUT,
            "owner": DecisionFailureOwner.PROVIDER,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PROVIDER_TIMEOUT,
        },
    ),
    (
        "network_failure",
        lambda: _base_observation(
            provider=ProviderQualificationSignal(network_failure=True),
        ),
        {
            "category": DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            "reason": DecisionFailureReason.NETWORK_FAILURE,
            "owner": DecisionFailureOwner.PROVIDER,
            "boundary": DecisionFailureBoundary.PROVIDER_BINDING,
            "retryability": DecisionRetryability.RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PROVIDER_NETWORK_FAILURE,
        },
    ),
    (
        "server_error",
        lambda: _base_observation(provider=ProviderQualificationSignal(server_error=True)),
        {
            "category": DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            "reason": DecisionFailureReason.PROVIDER_SERVER_ERROR,
            "owner": DecisionFailureOwner.PROVIDER,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PROVIDER_SERVER_ERROR,
        },
    ),
    (
        "protocol_error",
        lambda: _base_observation(
            provider=ProviderQualificationSignal(protocol_error=True),
        ),
        {
            "category": DecisionFailureCategory.PROVIDER_INFRASTRUCTURE,
            "reason": DecisionFailureReason.PROVIDER_PROTOCOL_ERROR,
            "owner": DecisionFailureOwner.PROVIDER,
            "boundary": DecisionFailureBoundary.PROVIDER_BINDING,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PROVIDER_PROTOCOL_ERROR,
        },
    ),
    (
        "wrong_execution_route",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                wrong_execution_route=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.WRONG_EXECUTION_ROUTE,
            "owner": DecisionFailureOwner.DECISION_SYSTEM,
            "boundary": DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_WRONG_EXECUTION_ROUTE,
        },
    ),
    (
        "trace_not_finalized",
        lambda: _base_observation(
            boundary=DecisionFailureBoundary.TRACE_FINALIZATION,
            platform_contract=PlatformContractQualificationSignal(trace_finalized=False),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.TRACE_RUN_NOT_FINALIZED,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.TRACE_FINALIZATION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_TRACE_NOT_FINALIZED,
        },
    ),
    (
        "strict_tool_violation",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                strict_tool_contract_violation=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.STRICT_TOOL_CONTRACT_VIOLATION,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_STRICT_TOOL_VIOLATION,
        },
    ),
    (
        "tool_dispatch_violation",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                tool_dispatch_contract_violation=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.TOOL_DISPATCH_CONTRACT_VIOLATION,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_TOOL_DISPATCH_VIOLATION,
        },
    ),
    (
        "invalid_phase_transition",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                invalid_phase_transition=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.INVALID_PHASE_TRANSITION,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_INVALID_PHASE_TRANSITION,
        },
    ),
    (
        "completion_reconciliation_violation",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                completion_reconciliation_contract_violation=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.COMPLETION_RECONCILIATION_CONTRACT_VIOLATION,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_COMPLETION_RECONCILIATION,
        },
    ),
    (
        "terminal_acceptance_violation",
        lambda: _base_observation(
            platform_contract=PlatformContractQualificationSignal(
                terminal_acceptance_contract_violation=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.PLATFORM_CONTRACT,
            "reason": DecisionFailureReason.TERMINAL_ACCEPTANCE_CONTRACT_VIOLATION,
            "owner": DecisionFailureOwner.EXECUTION_ENGINE,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.PLATFORM_TERMINAL_ACCEPTANCE,
        },
    ),
    (
        "insufficient_evidence",
        lambda: _base_observation(
            model_behavior=ModelBehaviorQualificationSignal(
                insufficient_evidence_gathering=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.MODEL_BEHAVIOR,
            "reason": DecisionFailureReason.INSUFFICIENT_EVIDENCE_GATHERING,
            "owner": DecisionFailureOwner.MODEL,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.MODEL_INSUFFICIENT_EVIDENCE,
        },
    ),
    (
        "tool_use_deficiency",
        lambda: _base_observation(
            model_behavior=ModelBehaviorQualificationSignal(tool_use_deficiency=True),
        ),
        {
            "category": DecisionFailureCategory.MODEL_BEHAVIOR,
            "reason": DecisionFailureReason.TOOL_USE_DEFICIENCY,
            "owner": DecisionFailureOwner.MODEL,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.MODEL_TOOL_USE_DEFICIENCY,
        },
    ),
    (
        "epistemic_contradiction",
        lambda: _base_observation(
            model_behavior=ModelBehaviorQualificationSignal(epistemic_contradiction=True),
        ),
        {
            "category": DecisionFailureCategory.MODEL_BEHAVIOR,
            "reason": DecisionFailureReason.EPISTEMIC_CONTRADICTION,
            "owner": DecisionFailureOwner.MODEL,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.MODEL_EPISTEMIC_CONTRADICTION,
        },
    ),
    (
        "unsupported_completion",
        lambda: _base_observation(
            model_behavior=ModelBehaviorQualificationSignal(unsupported_completion=True),
        ),
        {
            "category": DecisionFailureCategory.MODEL_BEHAVIOR,
            "reason": DecisionFailureReason.UNSUPPORTED_COMPLETION,
            "owner": DecisionFailureOwner.MODEL,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.MODEL_UNSUPPORTED_COMPLETION,
        },
    ),
    (
        "premature_completion",
        lambda: _base_observation(
            model_behavior=ModelBehaviorQualificationSignal(premature_completion=True),
        ),
        {
            "category": DecisionFailureCategory.MODEL_BEHAVIOR,
            "reason": DecisionFailureReason.PREMATURE_COMPLETION,
            "owner": DecisionFailureOwner.MODEL,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.MODEL_PREMATURE_COMPLETION,
        },
    ),
    (
        "evaluator_false_negative",
        lambda: _base_observation(
            evaluator=EvaluatorQualificationSignal(passed=False, false_negative=True),
        ),
        {
            "category": DecisionFailureCategory.EVALUATOR_SEMANTICS,
            "reason": DecisionFailureReason.FALSE_NEGATIVE,
            "owner": DecisionFailureOwner.EVALUATOR,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.EVALUATOR_FALSE_NEGATIVE,
        },
    ),
    (
        "evaluator_false_positive",
        lambda: _base_observation(
            evaluator=EvaluatorQualificationSignal(passed=False, false_positive=True),
        ),
        {
            "category": DecisionFailureCategory.EVALUATOR_SEMANTICS,
            "reason": DecisionFailureReason.FALSE_POSITIVE,
            "owner": DecisionFailureOwner.EVALUATOR,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.EVALUATOR_FALSE_POSITIVE,
        },
    ),
    (
        "criterion_semantics_invalid",
        lambda: _base_observation(
            evaluator=EvaluatorQualificationSignal(
                passed=False,
                criterion_semantics_invalid=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.EVALUATOR_SEMANTICS,
            "reason": DecisionFailureReason.CRITERION_SEMANTICS_INVALID,
            "owner": DecisionFailureOwner.EVALUATOR,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.EVALUATOR_CRITERION_SEMANTICS,
        },
    ),
    (
        "evaluator_contract_error",
        lambda: _base_observation(
            evaluator=EvaluatorQualificationSignal(passed=False, contract_error=True),
        ),
        {
            "category": DecisionFailureCategory.EVALUATOR_SEMANTICS,
            "reason": DecisionFailureReason.EVALUATOR_CONTRACT_ERROR,
            "owner": DecisionFailureOwner.EVALUATOR,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.NON_RETRIABLE,
            "diagnostic_code": DecisionFailureDiagnosticCode.EVALUATOR_CONTRACT_ERROR,
        },
    ),
    (
        "missing_signal",
        lambda: _base_observation(
            observability_complete=False,
            observability=ObservabilityQualificationSignal(missing_required_signal=True),
        ),
        {
            "category": DecisionFailureCategory.OBSERVABILITY_GAP,
            "reason": DecisionFailureReason.MISSING_REQUIRED_SIGNAL,
            "owner": DecisionFailureOwner.OBSERVABILITY,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.UNKNOWN,
            "diagnostic_code": DecisionFailureDiagnosticCode.OBSERVABILITY_MISSING_SIGNAL,
        },
    ),
    (
        "incomplete_trace",
        lambda: _base_observation(
            observability_complete=False,
            observability=ObservabilityQualificationSignal(incomplete_trace=True),
        ),
        {
            "category": DecisionFailureCategory.OBSERVABILITY_GAP,
            "reason": DecisionFailureReason.INCOMPLETE_TRACE,
            "owner": DecisionFailureOwner.OBSERVABILITY,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.UNKNOWN,
            "diagnostic_code": DecisionFailureDiagnosticCode.OBSERVABILITY_INCOMPLETE_TRACE,
        },
    ),
    (
        "ambiguous_boundary",
        lambda: _base_observation(
            observability_complete=False,
            observability=ObservabilityQualificationSignal(
                ambiguous_failure_boundary=True,
            ),
        ),
        {
            "category": DecisionFailureCategory.OBSERVABILITY_GAP,
            "reason": DecisionFailureReason.AMBIGUOUS_FAILURE_BOUNDARY,
            "owner": DecisionFailureOwner.OBSERVABILITY,
            "boundary": DecisionFailureBoundary.ROOT_CAPABILITY_ROUTING,
            "retryability": DecisionRetryability.UNKNOWN,
            "diagnostic_code": DecisionFailureDiagnosticCode.OBSERVABILITY_AMBIGUOUS_BOUNDARY,
        },
    ),
    (
        "unclassified",
        lambda: _base_observation(),
        {
            "category": DecisionFailureCategory.UNCLASSIFIED,
            "reason": DecisionFailureReason.UNCLASSIFIED,
            "owner": DecisionFailureOwner.DECISION_SYSTEM,
            "boundary": DecisionFailureBoundary.HOST_EXECUTION,
            "retryability": DecisionRetryability.UNKNOWN,
            "diagnostic_code": DecisionFailureDiagnosticCode.UNCLASSIFIED,
        },
    ),
)


@pytest.mark.parametrize(
    ("case_name", "observation_factory", "expected"),
    _GOLDEN_CASES,
    ids=[case[0] for case in _GOLDEN_CASES],
)
def test_golden_classification_matrix(
    case_name: str,
    observation_factory: Callable[[], DecisionQualificationObservation],
    expected: dict[str, object],
) -> None:
    _assert_classification(
        observation_factory(),
        category=expected["category"],  # type: ignore[arg-type]
        reason=expected["reason"],  # type: ignore[arg-type]
        owner=expected["owner"],  # type: ignore[arg-type]
        boundary=expected["boundary"],  # type: ignore[arg-type]
        retryability=expected["retryability"],  # type: ignore[arg-type]
        diagnostic_code=expected["diagnostic_code"],  # type: ignore[arg-type]
    )


def test_multiple_signal_precedence_prefers_environment() -> None:
    observation = _base_observation(
        environment=EnvironmentQualificationSignal(credential_unavailable=True),
        provider=ProviderQualificationSignal(rate_limit=True),
        model_behavior=ModelBehaviorQualificationSignal(tool_use_deficiency=True),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.ENVIRONMENT


def test_first_boundary_vs_category_precedence_prefers_model() -> None:
    observation = _base_observation(
        model_behavior=ModelBehaviorQualificationSignal(
            tool_use_deficiency=True,
            behavior_boundary=DecisionFailureBoundary.TOOL_DISPATCH,
        ),
        evaluator=EvaluatorQualificationSignal(passed=False, false_negative=True),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.category is DecisionFailureCategory.MODEL_BEHAVIOR
    assert result.reason is DecisionFailureReason.TOOL_USE_DEFICIENCY


def test_platform_violation_boundary_resolver_uses_explicit() -> None:
    observation = _base_observation(
        platform_contract=PlatformContractQualificationSignal(
            wrong_execution_route=True,
            violation_boundary=DecisionFailureBoundary.PHASE_VALIDATION,
        ),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.boundary is DecisionFailureBoundary.PHASE_VALIDATION


def test_model_behavior_boundary_resolver_uses_explicit() -> None:
    observation = _base_observation(
        model_behavior=ModelBehaviorQualificationSignal(
            epistemic_contradiction=True,
            behavior_boundary=DecisionFailureBoundary.REASONING,
        ),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.boundary is DecisionFailureBoundary.REASONING


def test_observation_boundary_earliest_with_default_resolver() -> None:
    observation = _base_observation(
        boundary=DecisionFailureBoundary.PROVIDER_BINDING,
        provider=ProviderQualificationSignal(timeout=True),
    )
    result = classify_decision_failure(observation)
    assert result is not None
    assert result.boundary is DecisionFailureBoundary.PROVIDER_BINDING


def test_golden_matrix_serialization_parity() -> None:
    for case_name, observation_factory, expected in _GOLDEN_CASES:
        result = classify_decision_failure(observation_factory())
        assert result is not None
        payload = classification_to_dict(result)
        assert payload["category"] == expected["category"].value  # type: ignore[union-attr]
        assert payload["reason"] == expected["reason"].value  # type: ignore[union-attr]
        assert payload["owner"] == expected["owner"].value  # type: ignore[union-attr]
        assert payload["boundary"] == expected["boundary"].value  # type: ignore[union-attr]
        assert payload["retryability"] == expected["retryability"].value  # type: ignore[union-attr]
        assert payload["diagnostic_code"] == expected["diagnostic_code"].value  # type: ignore[union-attr]


def test_all_diagnostic_codes_reachable() -> None:
    observed_codes = {
        classify_decision_failure(observation_factory())  # type: ignore[misc]
        .diagnostic_code  # type: ignore[union-attr]
        for _, observation_factory, _ in _GOLDEN_CASES
    }
    for code in DecisionFailureDiagnosticCode:
        assert code in observed_codes, f"diagnostic code not covered: {code}"


def test_all_failure_categories_reachable() -> None:
    observed_categories = {
        classify_decision_failure(observation_factory()).category  # type: ignore[union-attr]
        for _, observation_factory, _ in _GOLDEN_CASES
    }
    for category in DecisionFailureCategory:
        assert category in observed_categories, f"category not covered: {category}"


def test_all_failure_owners_reachable() -> None:
    observed_owners = {
        classify_decision_failure(observation_factory()).owner  # type: ignore[union-attr]
        for _, observation_factory, _ in _GOLDEN_CASES
    }
    for owner in DecisionFailureOwner:
        assert owner in observed_owners, f"owner not covered: {owner}"


def test_all_retryability_values_reachable() -> None:
    observed = {
        classify_decision_failure(observation_factory()).retryability  # type: ignore[union-attr]
        for _, observation_factory, _ in _GOLDEN_CASES
    }
    for retryability in DecisionRetryability:
        assert retryability in observed, f"retryability not covered: {retryability}"
