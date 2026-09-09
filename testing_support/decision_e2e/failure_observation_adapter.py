# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Scenario adapters mapping evaluator artifacts to qualification observations (DS-E2E-14.3)."""

from __future__ import annotations

from intergrax.decision_system.qualification.observation import DecisionQualificationObservation
from intergrax.decision_system.qualification.signals import (
    EnvironmentQualificationSignal,
    EvaluatorQualificationSignal,
    ModelBehaviorQualificationSignal,
    ObservabilityQualificationSignal,
    PlatformContractQualificationSignal,
    ProviderQualificationSignal,
)
from intergrax.decision_system.qualification.taxonomy import DecisionFailureBoundary

AI_INCIDENT_EPISTEMIC_FAILURE_ID = (
    "unsupported_inference:unresolved_with_supported_diagnosis"
)
AI_INCIDENT_TOOL_USE_FAILURE_ID = "tool_runtime_not_exercised"


def observation_from_ai_incident_evaluation(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
    trace_finalized: bool = True,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    """Map AI Incident scenario evaluator failures to structured observation facts."""
    model_behavior = ModelBehaviorQualificationSignal()
    platform_contract = PlatformContractQualificationSignal(trace_finalized=trace_finalized)

    if AI_INCIDENT_TOOL_USE_FAILURE_ID in failures:
        model_behavior = ModelBehaviorQualificationSignal(
            tool_use_deficiency=True,
            behavior_boundary=DecisionFailureBoundary.TOOL_DISPATCH,
        )
    if AI_INCIDENT_EPISTEMIC_FAILURE_ID in failures:
        model_behavior = ModelBehaviorQualificationSignal(
            epistemic_contradiction=True,
            behavior_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        )

    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=platform_contract,
        model_behavior=model_behavior,
        evaluator=EvaluatorQualificationSignal(passed=evaluator_passed),
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(),
        observability=ObservabilityQualificationSignal(),
    )


def observation_for_trace_not_finalized(
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.TRACE_FINALIZATION,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(trace_finalized=False),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(),
        observability=ObservabilityQualificationSignal(),
    )


def observation_for_provider_rate_limit(
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.PROVIDER_BINDING,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(rate_limit=True),
        environment=EnvironmentQualificationSignal(),
        observability=ObservabilityQualificationSignal(),
    )


def observation_for_credential_unavailable(
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(credential_unavailable=True),
        observability=ObservabilityQualificationSignal(),
    )
