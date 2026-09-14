# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Adapter from public ERL observation grouping SPI to central diagnostics (ERL-DIAG-001C-H)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.correlation import (
    ReliabilityDiagnosticCorrelation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.grouping import (
    ExternalEffectReliabilityProblemGroupingStrategy,
    parse_reliability_diagnostic_occurrence_instance_id,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId
from intergrax.runtime.diagnostics.reliability.observation_to_problem_signal import (
    ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID,
    ERL_RELIABILITY_SIGNAL_SOURCE_COMPONENT,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    ReliabilityDiagnosticHandoff,
)
from intergrax.runtime.diagnostics.reliability.reliability_observability_attributes import (
    ExternalEffectReliabilityObservabilityAttributes,
)
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY,
    PlatformProblemSignal,
)
from intergrax.runtime.observability.problem_signal import PlatformProblemSignal as _Signal


class ReliabilityObservationGroupingAdapterError(ValueError):
    """Observation or signal payload cannot support grouping strategy invocation."""


def grouping_subject_index_token_for_observation(
    observation: ExternalEffectReliabilityObservation,
    strategy: ExternalEffectReliabilityProblemGroupingStrategy,
) -> str:
    """Invoke the public grouping SPI and return the stable grouping index token."""
    subject = strategy.group(observation)
    if subject.tenant_id != observation.tenant_id:
        raise ReliabilityObservationGroupingAdapterError(
            "grouping strategy tenant_id does not match observation tenant_id",
        )
    token = subject.index_token
    if not token.strip():
        raise ReliabilityObservationGroupingAdapterError(
            "grouping strategy returned empty index_token",
        )
    return token


def grouping_subject_index_token_for_handoff(
    handoff: ReliabilityDiagnosticHandoff,
    strategy: ExternalEffectReliabilityProblemGroupingStrategy,
) -> str:
    return grouping_subject_index_token_for_observation(
        _observation_from_handoff(handoff),
        strategy,
    )


def grouping_subject_index_token_for_erl_signal_scope(
    *,
    tenant_id: str,
    application_id: str,
    instance_id: str,
    problem_signals: tuple[PlatformProblemSignal, ...],
    strategy: ExternalEffectReliabilityProblemGroupingStrategy,
) -> str:
    observation = observation_for_erl_signal_scope(
        tenant_id=tenant_id,
        application_id=application_id,
        instance_id=instance_id,
        problem_signals=problem_signals,
    )
    return grouping_subject_index_token_for_observation(observation, strategy)


def observation_for_erl_signal_scope(
    *,
    tenant_id: str,
    application_id: str,
    instance_id: str,
    problem_signals: tuple[PlatformProblemSignal, ...],
) -> ExternalEffectReliabilityObservation:
    if application_id != ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID:
        raise ReliabilityObservationGroupingAdapterError(
            "signal scope is not an ERL reliability diagnostic subject",
        )
    signal = _select_erl_reliability_signal(problem_signals)
    attrs = _erl_observability_attributes(signal)
    parsed_case_id, parsed_observation_id = parse_reliability_diagnostic_occurrence_instance_id(
        instance_id,
    )
    if attrs.observation_id != parsed_observation_id:
        raise ReliabilityObservationGroupingAdapterError(
            "signal observation_id does not match scope instance_id",
        )
    if attrs.reliability_case_id != parsed_case_id:
        raise ReliabilityObservationGroupingAdapterError(
            "signal reliability_case_id does not match scope instance_id case segment",
        )
    correlation = ReliabilityDiagnosticCorrelation(
        tenant_id=tenant_id,
        correlation_id=attrs.correlation_id,
        reliability_case_id=attrs.reliability_case_id,
        external_effect_contract_id=attrs.external_effect_contract_id,
        task_id=_optional_task_id(signal.task_id),
        run_id=_optional_run_id(signal.run_id),
        execution_id=_optional_execution_id(attrs.execution_id),
        attempt_id=_optional_attempt_id(attrs.attempt_id),
        trace_id=attrs.trace_id,
        idempotency_key=attrs.idempotency_key,
    )
    return ExternalEffectReliabilityObservation(
        observation_id=attrs.observation_id,
        tenant_id=tenant_id,
        signal_kind=ExternalEffectReliabilitySignalKind(attrs.signal_kind),
        recorded_at=signal.occurred_at,
        reliability_case_id=attrs.reliability_case_id,
        correlation=correlation,
        lifecycle_state=ReliabilityCaseLifecycleState(attrs.lifecycle_state),
        artifact_refs=ReliabilityDiagnosticArtifactRefs(),
        execution_safety_hint=AutomationSafetyHint(attrs.automation_safety_hint),
        trace_refs=attrs.trace_refs,
        source_transition_id=attrs.source_transition_id,
    )


def _observation_from_handoff(handoff: ReliabilityDiagnosticHandoff) -> ExternalEffectReliabilityObservation:
    return ExternalEffectReliabilityObservation(
        observation_id=handoff.observation_id,
        tenant_id=handoff.tenant_id,
        signal_kind=handoff.signal_kind,
        recorded_at=handoff.recorded_at,
        reliability_case_id=handoff.reliability_case_id,
        correlation=handoff.correlation,
        lifecycle_state=handoff.lifecycle_state,
        artifact_refs=handoff.artifact_refs,
        execution_safety_hint=handoff.automation_safety_hint,
        trace_refs=handoff.trace_refs,
        source_transition_id=handoff.source_transition_id,
    )


def _select_erl_reliability_signal(
    problem_signals: tuple[PlatformProblemSignal, ...],
) -> PlatformProblemSignal:
    if not problem_signals:
        raise ReliabilityObservationGroupingAdapterError(
            "ERL reliability signal scope requires at least one problem signal",
        )
    for signal in problem_signals:
        if type(signal) is not _Signal:
            raise TypeError("problem_signals entries must be PlatformProblemSignal")
        if signal.problem_kind != PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY:
            continue
        if signal.source_component != ERL_RELIABILITY_SIGNAL_SOURCE_COMPONENT:
            continue
        return signal
    raise ReliabilityObservationGroupingAdapterError(
        "no platform external effect reliability signal in scope",
    )


def _erl_observability_attributes(
    signal: PlatformProblemSignal,
) -> ExternalEffectReliabilityObservabilityAttributes:
    raw = signal.application_attributes
    if type(raw) is not ExternalEffectReliabilityObservabilityAttributes:
        raise ReliabilityObservationGroupingAdapterError(
            "ERL reliability signal missing typed observability attributes",
        )
    return raw


def _optional_task_id(value: str) -> TaskId | None:
    if not value.strip():
        return None
    return TaskId(value)


def _optional_run_id(value: str) -> RunId | None:
    if not value.strip():
        return None
    return RunId(value)


def _optional_execution_id(value: str | None) -> ExecutionId | None:
    if value is None or not value.strip():
        return None
    return ExecutionId(value)


def _optional_attempt_id(value: str | None) -> AttemptId | None:
    if value is None or not value.strip():
        return None
    return AttemptId(value)


__all__ = [
    "ReliabilityObservationGroupingAdapterError",
    "grouping_subject_index_token_for_erl_signal_scope",
    "grouping_subject_index_token_for_handoff",
    "grouping_subject_index_token_for_observation",
    "observation_for_erl_signal_scope",
]
