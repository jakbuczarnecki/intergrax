# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Deterministic mapping from ERL reliability handoff to PlatformProblemSignal (ERL-DIAG-001B)."""

from __future__ import annotations

from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    ExternalEffectReliabilitySignalKind,
)
from intergrax.runtime.diagnostics.reliability.reliability_diagnostic_handoff import (
    ReliabilityDiagnosticHandoff,
    ReliabilityDiagnosticHandoffIntegrityError,
)
from intergrax.runtime.diagnostics.reliability.reliability_observability_attributes import (
    ExternalEffectReliabilityObservabilityAttributes,
)
from intergrax.runtime.observability.export_attributes import ObservabilityArtifactReference
from intergrax.runtime.observability.problem_signal import (
    PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY,
    PROBLEM_SEVERITY_ERROR,
    PROBLEM_SEVERITY_INFO,
    PROBLEM_SEVERITY_WARNING,
    PROBLEM_SOURCE_LAYER_RUNTIME,
    PROBLEM_STATUS_DETECTED,
    PlatformProblemSignal,
)

ERL_RELIABILITY_SIGNAL_SOURCE_COMPONENT = "external_effect_reliability"
ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID = "erl"

def _neutral_safe_message(signal_kind: ExternalEffectReliabilitySignalKind) -> str:
    return f"External effect reliability signal: {signal_kind.value}"


def _conservative_severity(signal_kind: ExternalEffectReliabilitySignalKind) -> str:
    if signal_kind in (
        ExternalEffectReliabilitySignalKind.TRUTH_UNAVAILABLE,
        ExternalEffectReliabilitySignalKind.EVIDENCE_INSUFFICIENT,
        ExternalEffectReliabilitySignalKind.AUTOMATION_SAFETY_LIMIT,
    ):
        return PROBLEM_SEVERITY_ERROR
    if signal_kind in (
        ExternalEffectReliabilitySignalKind.UNCERTAINTY_ADMITTED,
        ExternalEffectReliabilitySignalKind.RECONCILIATION_ATTEMPTED,
    ):
        return PROBLEM_SEVERITY_WARNING
    return PROBLEM_SEVERITY_INFO


def _artifact_refs_to_observability(
    artifact_refs: ReliabilityDiagnosticArtifactRefs,
) -> tuple[ObservabilityArtifactReference, ...]:
    refs: list[ObservabilityArtifactReference] = []
    if artifact_refs.evidence_ref is not None and artifact_refs.evidence_ref.strip():
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.evidence_ref,
                schema_id="erl.diagnostics.evidence_ref",
            ),
        )
    if artifact_refs.reconciliation_ref is not None and artifact_refs.reconciliation_ref.strip():
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.reconciliation_ref,
                schema_id="erl.diagnostics.reconciliation_ref",
            ),
        )
    if artifact_refs.resolution_ref is not None and artifact_refs.resolution_ref.strip():
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.resolution_ref,
                schema_id="erl.diagnostics.resolution_ref",
            ),
        )
    if artifact_refs.governance_ref is not None and artifact_refs.governance_ref.strip():
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.governance_ref,
                schema_id="erl.diagnostics.governance_ref",
            ),
        )
    if artifact_refs.recovery_ref is not None and artifact_refs.recovery_ref.strip():
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.recovery_ref,
                schema_id="erl.diagnostics.recovery_ref",
            ),
        )
    if (
        artifact_refs.source_transition_ref is not None
        and artifact_refs.source_transition_ref.strip()
    ):
        refs.append(
            ObservabilityArtifactReference(
                artifact_ref=artifact_refs.source_transition_ref,
                schema_id="erl.diagnostics.source_transition_ref",
            ),
        )
    return tuple(refs)


def _build_observability_attributes(
    handoff: ReliabilityDiagnosticHandoff,
) -> ExternalEffectReliabilityObservabilityAttributes:
    correlation = handoff.correlation
    return ExternalEffectReliabilityObservabilityAttributes(
        observation_id=handoff.observation_id,
        signal_kind=handoff.signal_kind.value,
        reliability_case_id=handoff.reliability_case_id,
        lifecycle_state=handoff.lifecycle_state.value,
        automation_safety_hint=handoff.automation_safety_hint.value,
        external_effect_contract_id=correlation.external_effect_contract_id,
        correlation_id=correlation.correlation_id,
        execution_id=str(correlation.execution_id) if correlation.execution_id is not None else None,
        attempt_id=str(correlation.attempt_id) if correlation.attempt_id is not None else None,
        trace_id=correlation.trace_id,
        idempotency_key=correlation.idempotency_key,
        source_transition_id=handoff.source_transition_id,
        trace_refs=handoff.trace_refs,
    )


def map_handoff_to_platform_problem_signal(
    handoff: ReliabilityDiagnosticHandoff,
) -> PlatformProblemSignal:
    if type(handoff) is not ReliabilityDiagnosticHandoff:
        raise TypeError("handoff must be ReliabilityDiagnosticHandoff")
    if not handoff.observation_id.strip():
        raise ReliabilityDiagnosticHandoffIntegrityError("observation_id must be non-empty")

    correlation = handoff.correlation
    task_id = str(correlation.task_id) if correlation.task_id is not None else ""
    run_id = str(correlation.run_id) if correlation.run_id is not None else ""

    return PlatformProblemSignal(
        problem_kind=PROBLEM_KIND_PLATFORM_EXTERNAL_EFFECT_RELIABILITY,
        severity=_conservative_severity(handoff.signal_kind),
        source_layer=PROBLEM_SOURCE_LAYER_RUNTIME,
        source_component=ERL_RELIABILITY_SIGNAL_SOURCE_COMPONENT,
        status=PROBLEM_STATUS_DETECTED,
        safe_message=_neutral_safe_message(handoff.signal_kind),
        error_code=handoff.observation_id,
        exception_type=None,
        run_id=run_id,
        task_id=task_id,
        event_id=handoff.observation_id,
        correlation_id=correlation.correlation_id,
        occurred_at=handoff.recorded_at,
        application_attributes=_build_observability_attributes(handoff),
        artifact_refs=_artifact_refs_to_observability(handoff.artifact_refs),
    )


__all__ = [
    "ERL_RELIABILITY_DIAGNOSTIC_SUBJECT_APPLICATION_ID",
    "ERL_RELIABILITY_SIGNAL_SOURCE_COMPONENT",
    "map_handoff_to_platform_problem_signal",
]
