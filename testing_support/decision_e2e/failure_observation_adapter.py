# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Scenario adapters mapping evaluator artifacts to qualification observations (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

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
from testing_support.decision_e2e.ai_incident_failure_qualification_mapping import (
    AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID,
    AI_INCIDENT_EPISTEMIC_FAILURE_ID,
    AI_INCIDENT_EPISTEMIC_FAILURE_IDS,
    AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS,
    AI_INCIDENT_REVISION_FLOW_FAILURE_IDS,
    AiIncidentFailureMappingError,
    AiIncidentQualificationInputError,
    build_ai_incident_qualification_signals,
)

__all__ = (
    "AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID",
    "AI_INCIDENT_EPISTEMIC_FAILURE_ID",
    "AI_INCIDENT_EPISTEMIC_FAILURE_IDS",
    "AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS",
    "AI_INCIDENT_REVISION_FLOW_FAILURE_IDS",
    "AiIncidentFailureMappingError",
    "AiIncidentQualificationInputError",
    "EnvironmentQualificationFacts",
    "ProviderInfrastructureFacts",
    "observation_for_credential_unavailable",
    "observation_for_provider_rate_limit",
    "observation_for_trace_not_finalized",
    "observation_from_ai_incident_evaluation",
    "observation_from_environment_facts",
    "observation_from_platform_trace_readback",
    "observation_from_provider_infrastructure_facts",
    "observation_from_scenario_execution_exception",
    "provider_infrastructure_facts_from_execution_error",
)


def observation_from_ai_incident_evaluation(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
    trace_finalized: bool = True,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    """Map AI Incident scenario evaluator failures to structured observation facts."""
    (
        model_behavior,
        platform_contract,
        evaluator,
        observability,
        effective_boundary,
    ) = build_ai_incident_qualification_signals(
        failures=failures,
        evaluator_passed=evaluator_passed,
        trace_finalized=trace_finalized,
        boundary=boundary,
    )

    return DecisionQualificationObservation(
        boundary=effective_boundary,
        platform_contract=platform_contract,
        model_behavior=model_behavior,
        evaluator=evaluator,
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(),
        observability=observability,
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


@dataclass(frozen=True, slots=True)
class EnvironmentQualificationFacts:
    credential_unavailable: bool = False
    provider_configuration_invalid: bool = False
    model_configuration_invalid: bool = False
    qualification_disabled: bool = False


@dataclass(frozen=True, slots=True)
class ProviderInfrastructureFacts:
    rate_limit: bool = False
    timeout: bool = False
    network_failure: bool = False
    server_error: bool = False
    protocol_error: bool = False


def observation_from_environment_facts(
    facts: EnvironmentQualificationFacts,
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.ENVIRONMENT_RESOLUTION,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(
            credential_unavailable=facts.credential_unavailable,
            provider_configuration_invalid=facts.provider_configuration_invalid,
            model_configuration_invalid=facts.model_configuration_invalid,
            qualification_disabled=facts.qualification_disabled,
        ),
        observability=ObservabilityQualificationSignal(),
    )


def observation_from_provider_infrastructure_facts(
    facts: ProviderInfrastructureFacts,
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(
            rate_limit=facts.rate_limit,
            timeout=facts.timeout,
            network_failure=facts.network_failure,
            server_error=facts.server_error,
            protocol_error=facts.protocol_error,
        ),
        environment=EnvironmentQualificationSignal(),
        observability=ObservabilityQualificationSignal(),
    )


def observation_from_platform_trace_readback(
    *,
    trace_finalized: bool,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.TRACE_FINALIZATION,
) -> DecisionQualificationObservation:
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(trace_finalized=trace_finalized),
        model_behavior=ModelBehaviorQualificationSignal(),
        evaluator=EvaluatorQualificationSignal(passed=False),
        provider=ProviderQualificationSignal(),
        environment=EnvironmentQualificationSignal(),
        observability=ObservabilityQualificationSignal(),
    )


def observation_from_scenario_execution_exception(
    exc: BaseException,
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    """Map structured scenario execution exceptions to qualification observations."""
    from testing_support.decision_e2e.scenario_exception_qualification import (
        observation_from_scenario_execution_exception as _map_scenario_exception,
    )

    return _map_scenario_exception(exc, boundary=boundary)


def provider_infrastructure_facts_from_execution_error(
    *,
    exc: BaseException,
    http_status: int | None = None,
) -> ProviderInfrastructureFacts | None:
    """Map structured execution error facts to provider infrastructure signals."""
    from testing_support.decision_e2e.scenario_exception_qualification import (
        provider_infrastructure_facts_from_execution_error as _map_provider_facts,
    )

    return _map_provider_facts(exc=exc, http_status=http_status)
