# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Typed internal handoff from public ERL observation to central diagnostics (ERL-DIAG-001B)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from intergrax.contracts.enterprise_reliability.case_lifecycle import (
    ReliabilityCaseLifecycleState,
)
from intergrax.contracts.enterprise_reliability.diagnostics.artifact_refs import (
    ReliabilityDiagnosticArtifactRefs,
)
from intergrax.contracts.enterprise_reliability.diagnostics.correlation import (
    ReliabilityDiagnosticCorrelation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.observation import (
    ExternalEffectReliabilityObservation,
)
from intergrax.contracts.enterprise_reliability.diagnostics.taxonomy import (
    AutomationSafetyHint,
    ExternalEffectReliabilitySignalKind,
)
from intergrax.contracts.execution_identity import AttemptId, ExecutionId, RunId, TaskId


class ReliabilityDiagnosticHandoffIntegrityError(ValueError):
    """Observation could not be mapped into a typed diagnostics handoff."""


@dataclass(frozen=True, slots=True)
class ReliabilityDiagnosticHandoff:
    """
    Immutable bridge-local view of one accepted observation.

    Preserves distinct identity domains required for downstream grouping strategies.
    """

    observation_id: str
    tenant_id: str
    signal_kind: ExternalEffectReliabilitySignalKind
    recorded_at: datetime
    reliability_case_id: str
    correlation: ReliabilityDiagnosticCorrelation
    lifecycle_state: ReliabilityCaseLifecycleState
    artifact_refs: ReliabilityDiagnosticArtifactRefs
    automation_safety_hint: AutomationSafetyHint
    trace_refs: tuple[str, ...]
    source_transition_id: str | None


def map_observation_to_handoff(
    observation: ExternalEffectReliabilityObservation,
) -> ReliabilityDiagnosticHandoff:
    if type(observation) is not ExternalEffectReliabilityObservation:
        raise TypeError("observation must be ExternalEffectReliabilityObservation")
    return ReliabilityDiagnosticHandoff(
        observation_id=observation.observation_id,
        tenant_id=observation.tenant_id,
        signal_kind=observation.signal_kind,
        recorded_at=observation.recorded_at,
        reliability_case_id=observation.reliability_case_id,
        correlation=observation.correlation,
        lifecycle_state=observation.lifecycle_state,
        artifact_refs=observation.artifact_refs,
        automation_safety_hint=observation.execution_safety_hint,
        trace_refs=observation.trace_refs,
        source_transition_id=observation.source_transition_id,
    )


def handoff_execution_ids(
    handoff: ReliabilityDiagnosticHandoff,
) -> tuple[TaskId | None, RunId | None, ExecutionId | None, AttemptId | None]:
    correlation = handoff.correlation
    return (
        correlation.task_id,
        correlation.run_id,
        correlation.execution_id,
        correlation.attempt_id,
    )


__all__ = [
    "ReliabilityDiagnosticHandoff",
    "ReliabilityDiagnosticHandoffIntegrityError",
    "handoff_execution_ids",
    "map_observation_to_handoff",
]
