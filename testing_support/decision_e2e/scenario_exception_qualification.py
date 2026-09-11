# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed scenario execution exception → qualification observation mapping (DS-E2E-15J-C1)."""

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
from testing_support.decision_e2e.failure_observation_adapter import (
    ProviderInfrastructureFacts,
    observation_from_provider_infrastructure_facts,
)


def provider_infrastructure_facts_from_execution_error(
    *,
    exc: BaseException,
    http_status: int | None = None,
) -> ProviderInfrastructureFacts | None:
    """Map structured execution error facts to provider infrastructure signals.

    Returns ``None`` when there is no positive provider-infrastructure evidence.
    """
    if http_status == 429:
        return ProviderInfrastructureFacts(rate_limit=True)
    if http_status is not None and 500 <= http_status <= 599:
        return ProviderInfrastructureFacts(server_error=True)
    if isinstance(exc, TimeoutError):
        return ProviderInfrastructureFacts(timeout=True)
    if isinstance(exc, ConnectionError):
        return ProviderInfrastructureFacts(network_failure=True)

    error_type = type(exc).__name__
    lowered = error_type.lower()
    if "timeout" in lowered:
        return ProviderInfrastructureFacts(timeout=True)
    if any(token in lowered for token in ("connection", "network", "unreachable")):
        return ProviderInfrastructureFacts(network_failure=True)
    if any(token in lowered for token in ("protocol", "parse")):
        return ProviderInfrastructureFacts(protocol_error=True)
    if http_status is not None and http_status >= 400:
        return ProviderInfrastructureFacts(protocol_error=True)
    return None


def observation_for_unclassified_scenario_exception(
    *,
    boundary: DecisionFailureBoundary = DecisionFailureBoundary.HOST_EXECUTION,
) -> DecisionQualificationObservation:
    """Fail-closed observation when no causal mapping exists for an execution exception."""
    return DecisionQualificationObservation(
        boundary=boundary,
        platform_contract=PlatformContractQualificationSignal(),
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
    from intergrax.decision_system.completion_eligibility import CompletionEligibilityStatus
    from intergrax.llm_adapters.contracts.strict_tool_call_validation import (
        StrictToolContractValidationError,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.completion_reconciliation import (
        CompletionReconciliationError,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.completion_transition import (
        PreReconciliationValidationError,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.evidence_completion_gate import (
        CompletionEligibilityBlockedError,
    )
    from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
        TERMINAL_STATE_NOT_ACCEPTED,
    )

    if isinstance(exc, PreReconciliationValidationError):
        return DecisionQualificationObservation(
            boundary=DecisionFailureBoundary.PHASE_VALIDATION,
            platform_contract=PlatformContractQualificationSignal(trace_finalized=True),
            model_behavior=ModelBehaviorQualificationSignal(
                unsupported_completion=True,
                behavior_boundary=DecisionFailureBoundary.PHASE_VALIDATION,
            ),
            evaluator=EvaluatorQualificationSignal(passed=False),
            provider=ProviderQualificationSignal(),
            environment=EnvironmentQualificationSignal(),
            observability=ObservabilityQualificationSignal(),
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

    if isinstance(exc, CompletionEligibilityBlockedError):
        decision = exc.decision
        if (
            decision.status is CompletionEligibilityStatus.INELIGIBLE
            and decision.unresolved_mandatory_requirement_ids
        ):
            return DecisionQualificationObservation(
                boundary=DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
                platform_contract=PlatformContractQualificationSignal(trace_finalized=True),
                model_behavior=ModelBehaviorQualificationSignal(
                    insufficient_evidence_gathering=True,
                    behavior_boundary=DecisionFailureBoundary.EVIDENCE_LIFECYCLE,
                ),
                evaluator=EvaluatorQualificationSignal(passed=False),
                provider=ProviderQualificationSignal(),
                environment=EnvironmentQualificationSignal(),
                observability=ObservabilityQualificationSignal(),
            )
        return observation_for_unclassified_scenario_exception(boundary=boundary)

    if isinstance(exc, StrictToolContractValidationError):
        return DecisionQualificationObservation(
            boundary=DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
            platform_contract=PlatformContractQualificationSignal(
                trace_finalized=True,
                strict_tool_contract_violation=True,
                violation_boundary=DecisionFailureBoundary.STRICT_TOOL_PROJECTION,
            ),
            model_behavior=ModelBehaviorQualificationSignal(),
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

    facts = provider_infrastructure_facts_from_execution_error(exc=exc)
    if facts is not None:
        return observation_from_provider_infrastructure_facts(facts, boundary=boundary)

    return observation_for_unclassified_scenario_exception(boundary=boundary)
