# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Evidence-backed operator diagnostic assessment (DIAG-4)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.contracts.execution_identity import AttemptId, EventId, RunId, TaskId
from intergrax.runtime.diagnostics.execution_reconstruction import ExecutionReconstruction
from intergrax.runtime.diagnostics.lifecycle_analysis import (
    LifecycleAnalysis,
    LifecycleAnomaly,
    LifecycleAnomalyKind,
    LifecycleAnomalyScope,
)
from intergrax.runtime.events.execution_position import ExecutionEventPosition


class DiagnosticAssessmentIntegrityError(Exception):
    """Raised when reconstruction and lifecycle analysis scopes do not match."""


class DiagnosticCertainty(StrEnum):
    """Semantic certainty for operator-facing diagnostic claims."""

    PROVEN = "proven"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DiagnosticFindingKind(StrEnum):
    """Operator-facing conclusion kinds derived from lifecycle anomalies."""

    CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY = "causal_attempt_without_runtime_history"
    RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE = "runtime_attempt_without_causal_evidence"
    MULTIPLE_TERMINAL_OUTCOMES = "multiple_terminal_outcomes"
    EVENT_AFTER_TERMINAL = "event_after_terminal"
    DISALLOWED_AFTER_FAILED = "disallowed_after_failed"


class DiagnosticLimitationKind(StrEnum):
    """Facts that constrain what stronger conclusions can be proven."""

    RUNTIME_HISTORY_TRUNCATED = "runtime_history_truncated"


_ANOMALY_TO_FINDING_KIND: dict[LifecycleAnomalyKind, DiagnosticFindingKind] = {
    LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY: (
        DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY
    ),
    LifecycleAnomalyKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE: (
        DiagnosticFindingKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE
    ),
    LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES: (
        DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES
    ),
    LifecycleAnomalyKind.EVENT_AFTER_TERMINAL: DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
    LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED: DiagnosticFindingKind.DISALLOWED_AFTER_FAILED,
}

_PROVEN_CLAIMS: dict[DiagnosticFindingKind, str] = {
    DiagnosticFindingKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY: (
        "Transport-to-execution causal evidence exists for this attempt, but no "
        "RuntimeEvent history for the attempt is present in the reconstruction."
    ),
    DiagnosticFindingKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE: (
        "This execution contains transport-origin evidence, but this runtime attempt "
        "has no corresponding transport→execution causal evidence."
    ),
    DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES: (
        "Canonical lifecycle facts contain conflicting final outcomes."
    ),
    DiagnosticFindingKind.EVENT_AFTER_TERMINAL: (
        "A lifecycle event was recorded after canonical run closure "
        "(COMPLETED or CANCELLED)."
    ),
    DiagnosticFindingKind.DISALLOWED_AFTER_FAILED: (
        "While execution was in FAILED state, a lifecycle transition occurred that "
        "the canonical lifecycle contract does not permit before a valid retry."
    ),
}

_LIMITATION_MESSAGES: dict[DiagnosticLimitationKind, str] = {
    DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED: (
        "Runtime history is truncated; conclusions requiring the unseen tail "
        "cannot be proven."
    ),
}

# Exhaustive mapping contract: every LifecycleAnomalyKind must map to finding or limitation.
_ANOMALY_OUTPUT_KIND: dict[LifecycleAnomalyKind, str] = {
    LifecycleAnomalyKind.CAUSAL_ATTEMPT_WITHOUT_RUNTIME_HISTORY: "finding",
    LifecycleAnomalyKind.RUNTIME_ATTEMPT_WITHOUT_CAUSAL_EVIDENCE: "finding",
    LifecycleAnomalyKind.RUNTIME_HISTORY_TRUNCATED: "limitation",
    LifecycleAnomalyKind.MULTIPLE_TERMINAL_OUTCOMES: "finding",
    LifecycleAnomalyKind.EVENT_AFTER_TERMINAL: "finding",
    LifecycleAnomalyKind.DISALLOWED_AFTER_FAILED: "finding",
}


@dataclass(frozen=True, slots=True)
class DiagnosticFinding:
    """One evidence-backed operator conclusion derived from a lifecycle anomaly."""

    kind: DiagnosticFindingKind
    scope: LifecycleAnomalyScope
    attempt_id: AttemptId | None
    certainty: DiagnosticCertainty
    claim: str
    source_anomaly_kind: LifecycleAnomalyKind
    supporting_event_ids: tuple[EventId, ...]
    supporting_evidence_ids: tuple[EventId, ...]
    supporting_positions: tuple[ExecutionEventPosition, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticLimitation:
    """A factual constraint on what stronger conclusions can be proven."""

    kind: DiagnosticLimitationKind
    factual_message: str
    source_anomaly_kind: LifecycleAnomalyKind
    supporting_event_ids: tuple[EventId, ...]
    supporting_evidence_ids: tuple[EventId, ...]
    supporting_positions: tuple[ExecutionEventPosition, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticAssessment:
    """
    Derived operator-facing assessment for one execution scope.

    NOT persisted and NOT a source of truth.
    """

    tenant_id: str
    task_id: TaskId
    run_id: RunId
    findings: tuple[DiagnosticFinding, ...]
    limitations: tuple[DiagnosticLimitation, ...]

    @property
    def has_findings(self) -> bool:
        return bool(self.findings)

    @property
    def has_limitations(self) -> bool:
        return bool(self.limitations)


class DiagnosticAssessmentBuilder:
    """
    Deterministic operator assessment over reconstruction + lifecycle analysis.

    Does not query persistence, emit events, infer root cause, or use LLM.
    """

    def assess(
        self,
        reconstruction: ExecutionReconstruction,
        lifecycle: LifecycleAnalysis,
    ) -> DiagnosticAssessment:
        _validate_assessment_scope(reconstruction, lifecycle)

        findings: list[DiagnosticFinding] = []
        limitations: list[DiagnosticLimitation] = []

        for anomaly in lifecycle.anomalies:
            output_kind = _ANOMALY_OUTPUT_KIND[anomaly.kind]
            if output_kind == "limitation":
                limitations.append(_limitation_from_anomaly(anomaly))
            else:
                findings.append(_finding_from_anomaly(anomaly))

        return DiagnosticAssessment(
            tenant_id=reconstruction.tenant_id,
            task_id=reconstruction.task_id,
            run_id=reconstruction.run_id,
            findings=tuple(findings),
            limitations=tuple(limitations),
        )


def _validate_assessment_scope(
    reconstruction: ExecutionReconstruction,
    lifecycle: LifecycleAnalysis,
) -> None:
    if reconstruction.tenant_id != lifecycle.tenant_id:
        raise DiagnosticAssessmentIntegrityError(
            "reconstruction tenant_id does not match lifecycle analysis scope"
        )
    if reconstruction.task_id != lifecycle.task_id:
        raise DiagnosticAssessmentIntegrityError(
            "reconstruction task_id does not match lifecycle analysis scope"
        )
    if reconstruction.run_id != lifecycle.run_id:
        raise DiagnosticAssessmentIntegrityError(
            "reconstruction run_id does not match lifecycle analysis scope"
        )


def _finding_from_anomaly(anomaly: LifecycleAnomaly) -> DiagnosticFinding:
    finding_kind = _ANOMALY_TO_FINDING_KIND[anomaly.kind]
    return DiagnosticFinding(
        kind=finding_kind,
        scope=anomaly.scope,
        attempt_id=anomaly.attempt_id,
        certainty=DiagnosticCertainty.PROVEN,
        claim=_PROVEN_CLAIMS[finding_kind],
        source_anomaly_kind=anomaly.kind,
        supporting_event_ids=anomaly.supporting_event_ids,
        supporting_evidence_ids=anomaly.supporting_evidence_ids,
        supporting_positions=anomaly.supporting_positions,
    )


def _limitation_from_anomaly(anomaly: LifecycleAnomaly) -> DiagnosticLimitation:
    limitation_kind = DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED
    return DiagnosticLimitation(
        kind=limitation_kind,
        factual_message=_LIMITATION_MESSAGES[limitation_kind],
        source_anomaly_kind=anomaly.kind,
        supporting_event_ids=anomaly.supporting_event_ids,
        supporting_evidence_ids=anomaly.supporting_evidence_ids,
        supporting_positions=anomaly.supporting_positions,
    )
