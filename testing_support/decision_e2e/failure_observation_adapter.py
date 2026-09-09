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

AI_INCIDENT_EPISTEMIC_FAILURE_ID = (
    "unsupported_inference:unresolved_with_supported_diagnosis"
)
AI_INCIDENT_DIAGNOSTIC_TOOL_TRACE_FAILURE_ID = "tool_runtime_not_exercised"

AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS: frozenset[str] = frozenset(
    {
        "staffing_attendance_not_gathered",
        "staffing_preliminary_not_gathered",
        "comparison_evidence_not_gathered",
        "telemetry_evidence_not_in_graph",
    }
)
AI_INCIDENT_REVISION_FLOW_FAILURE_IDS: frozenset[str] = frozenset(
    {
        "telemetry_visible_before_revision",
        "critic_falsification_missing",
        "failed_critic_verdict_missing",
        "evidence_challenge_missing",
        "bounded_recovery_missing",
        "follow_up_not_via_tools",
    }
)
AI_INCIDENT_EPISTEMIC_FAILURE_IDS: frozenset[str] = frozenset(
    {
        AI_INCIDENT_EPISTEMIC_FAILURE_ID,
        "h2_not_rejected",
    }
)


def _model_behavior_from_ai_incident_failures(
    failures: tuple[str, ...],
) -> ModelBehaviorQualificationSignal:
    failure_set = frozenset(failures)
    if failure_set.intersection(AI_INCIDENT_INSUFFICIENT_EVIDENCE_FAILURE_IDS):
        return ModelBehaviorQualificationSignal(
            insufficient_evidence_gathering=True,
            behavior_boundary=DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
        )
    if failure_set.intersection(AI_INCIDENT_REVISION_FLOW_FAILURE_IDS):
        return ModelBehaviorQualificationSignal(
            premature_completion=True,
            behavior_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        )
    if failure_set.intersection(AI_INCIDENT_EPISTEMIC_FAILURE_IDS):
        return ModelBehaviorQualificationSignal(
            epistemic_contradiction=True,
            behavior_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
        )
    return ModelBehaviorQualificationSignal()


def observation_from_ai_incident_evaluation(
    *,
    failures: tuple[str, ...],
    evaluator_passed: bool,
    trace_finalized: bool = True,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    """Map AI Incident scenario evaluator failures to structured observation facts."""
    model_behavior = _model_behavior_from_ai_incident_failures(failures)
    platform_contract = PlatformContractQualificationSignal(trace_finalized=trace_finalized)

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
    from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
        CompletionReconciliationError,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
        TERMINAL_STATE_NOT_ACCEPTED,
    )

    if isinstance(exc, CompletionReconciliationError):
        return DecisionQualificationObservation(
            boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
            platform_contract=PlatformContractQualificationSignal(trace_finalized=True),
            model_behavior=ModelBehaviorQualificationSignal(
                unsupported_completion=True,
                behavior_boundary=DecisionFailureBoundary.COMPLETION_RECONCILIATION,
            ),
            evaluator=EvaluatorQualificationSignal(passed=False),
            provider=ProviderQualificationSignal(),
            environment=EnvironmentQualificationSignal(),
            observability=ObservabilityQualificationSignal(),
        )

    if isinstance(exc, RuntimeError) and TERMINAL_STATE_NOT_ACCEPTED in str(exc):
        return DecisionQualificationObservation(
            boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
            platform_contract=PlatformContractQualificationSignal(
                terminal_acceptance_contract_violation=True,
                violation_boundary=DecisionFailureBoundary.TERMINAL_ACCEPTANCE,
            ),
            model_behavior=ModelBehaviorQualificationSignal(),
            evaluator=EvaluatorQualificationSignal(passed=False),
            provider=ProviderQualificationSignal(),
            environment=EnvironmentQualificationSignal(),
            observability=ObservabilityQualificationSignal(),
        )

    facts = provider_infrastructure_facts_from_execution_error(
        error_type=type(exc).__name__,
    )
    return observation_from_provider_infrastructure_facts(facts, boundary=boundary)


def provider_infrastructure_facts_from_execution_error(
    *,
    error_type: str,
    http_status: int | None = None,
) -> ProviderInfrastructureFacts:
    """Map structured execution error facts to provider infrastructure signals."""
    if http_status == 429:
        return ProviderInfrastructureFacts(rate_limit=True)
    if http_status is not None and 500 <= http_status <= 599:
        return ProviderInfrastructureFacts(server_error=True)
    lowered = error_type.lower()
    if "timeout" in lowered:
        return ProviderInfrastructureFacts(timeout=True)
    if any(token in lowered for token in ("connection", "network", "unreachable")):
        return ProviderInfrastructureFacts(network_failure=True)
    if any(token in lowered for token in ("protocol", "parse", "json")):
        return ProviderInfrastructureFacts(protocol_error=True)
    if http_status is not None and http_status >= 400:
        return ProviderInfrastructureFacts(protocol_error=True)
    return ProviderInfrastructureFacts(network_failure=True)
