# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""DG-003 deterministic last-good → first-failed operator story projection."""

from __future__ import annotations

from intergrax.contracts.execution_event_position import ExecutionEventPosition
from intergrax.contracts.execution_identity import EventId
from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.execution_reconstruction_models import (
    RuntimeHistoryCompleteness,
)
from intergrax.contracts.positioned_runtime_event import PositionedRuntimeEvent
from intergrax.contracts.runtime_event_type import RuntimeEventType
from intergrax.runtime.diagnostics.diagnostic_assessment import (
    DiagnosticAssessment,
    DiagnosticFinding,
    DiagnosticFindingKind,
    DiagnosticLimitationKind,
)
from intergrax.runtime.diagnostics.diagnostic_operator_investigation_read_models import (
    DiagnosticOperatorStory,
    DiagnosticOperatorStoryEvidenceRef,
    DiagnosticOperatorStoryFirstFailedScope,
    DiagnosticOperatorStoryPoint,
    DiagnosticOperatorStoryPointStatus,
    DiagnosticOperatorStoryTransition,
)
from intergrax.runtime.diagnostics.diagnostic_read_models import (
    DiagnosticOccurrenceReadStatus,
    DiagnosticProblemOccurrenceView,
)

_STORY_FAILURE_FINDING_KINDS: frozenset[DiagnosticFindingKind] = frozenset(
    {
        DiagnosticFindingKind.EXECUTION_FAILED,
        DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED,
        DiagnosticFindingKind.DISALLOWED_AFTER_FAILED,
        DiagnosticFindingKind.EVENT_AFTER_TERMINAL,
        DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES,
    }
)

_FAILURE_RUNTIME_EVENT_TYPES: frozenset[RuntimeEventType] = frozenset(
    {
        RuntimeEventType.EXECUTION_FAILED,
        RuntimeEventType.EXTERNAL_OPERATION_FAILED,
    }
)

_FINDING_KIND_ORDER: dict[DiagnosticFindingKind, int] = {
    DiagnosticFindingKind.EXECUTION_FAILED: 0,
    DiagnosticFindingKind.EXTERNAL_OPERATION_FAILED: 1,
    DiagnosticFindingKind.DISALLOWED_AFTER_FAILED: 2,
    DiagnosticFindingKind.EVENT_AFTER_TERMINAL: 3,
    DiagnosticFindingKind.MULTIPLE_TERMINAL_OUTCOMES: 4,
}


class DiagnosticOperatorStoryIntegrityError(Exception):
    """Raised when operator story anchors violate canonical position ordering."""


def project_operator_story(
    *,
    occurrence: DiagnosticProblemOccurrenceView,
    reconstruction: ExecutionReconstruction | None,
    assessment: DiagnosticAssessment | None,
) -> DiagnosticOperatorStory:
    """
    Pure deterministic projection over reconstruction + assessment.

    ExecutionEventPosition is authoritative; timestamps are display-only.
    """
    if reconstruction is None:
        reason = "no_execution_subject"
        if occurrence.read_status is not DiagnosticOccurrenceReadStatus.AVAILABLE:
            reason = occurrence.unavailable_reason or "occurrence_unavailable"
        return _unavailable_story(reason)

    limitations: list[str] = []
    history_truncated = (
        reconstruction.runtime_history_completeness
        is RuntimeHistoryCompleteness.TRUNCATED
    )
    if history_truncated:
        limitations.append(
            "Runtime history is truncated; first failure is only first observed "
            "in available evidence."
        )
    if assessment is not None:
        for limitation in assessment.limitations:
            if limitation.kind is DiagnosticLimitationKind.RUNTIME_HISTORY_TRUNCATED:
                if not history_truncated:
                    limitations.append(limitation.factual_message)

    positioned = _sorted_positioned_events(reconstruction.positioned_events)
    if assessment is None:
        return DiagnosticOperatorStory(
            last_good=_unavailable_point("no_diagnostic_assessment"),
            first_failed=_unavailable_point("no_diagnostic_assessment"),
            transition=None,
            supporting_evidence=(),
            limitations=tuple(limitations),
            first_failed_scope=None,
        )

    failure_findings = [
        finding
        for finding in assessment.findings
        if finding.kind in _STORY_FAILURE_FINDING_KINDS
    ]
    if not failure_findings:
        return DiagnosticOperatorStory(
            last_good=_unavailable_point("no_factual_failure_in_available_evidence"),
            first_failed=_unavailable_point("no_factual_failure_in_available_evidence"),
            transition=None,
            supporting_evidence=(),
            limitations=tuple(limitations),
            first_failed_scope=None,
        )

    primary = _select_first_failure_finding(failure_findings, positioned)
    failure_position = _failure_position_for_finding(primary, positioned)
    if failure_position is None:
        return DiagnosticOperatorStory(
            last_good=_unavailable_point("failure_boundary_position_unavailable"),
            first_failed=_unavailable_point("failure_boundary_position_unavailable"),
            transition=None,
            supporting_evidence=_evidence_refs_for_finding(primary),
            limitations=tuple(limitations),
            first_failed_scope=None,
        )

    failure_row = _row_at_position(positioned, failure_position)
    first_failed = _point_from_row(
        failure_row,
        finding_kind=primary.kind,
    )

    last_good_row = _select_last_good_predecessor(positioned, failure_position)
    if last_good_row is None:
        last_good = _unavailable_point("no_prior_factual_event_available")
    else:
        last_good = _point_from_row(last_good_row, finding_kind=None)

    if (
        last_good.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
        and first_failed.status is DiagnosticOperatorStoryPointStatus.AVAILABLE
        and last_good.position is not None
        and first_failed.position is not None
        and last_good.position.value >= first_failed.position.value
    ):
        raise DiagnosticOperatorStoryIntegrityError(
            "last_good position must be strictly before first_failed position"
        )

    transition = _build_transition(last_good, first_failed, positioned)
    evidence = _dedupe_evidence_refs(
        _evidence_refs_for_finding(primary)
        + _evidence_refs_for_row(last_good_row)
        + _evidence_refs_for_row(failure_row)
    )

    scope = (
        DiagnosticOperatorStoryFirstFailedScope.FIRST_OBSERVED_IN_AVAILABLE_EVIDENCE
        if history_truncated
        else DiagnosticOperatorStoryFirstFailedScope.PROVEN_IN_AVAILABLE_EVIDENCE
    )

    return DiagnosticOperatorStory(
        last_good=last_good,
        first_failed=first_failed,
        transition=transition,
        supporting_evidence=evidence,
        limitations=tuple(limitations),
        first_failed_scope=scope,
    )


def _unavailable_story(reason: str) -> DiagnosticOperatorStory:
    return DiagnosticOperatorStory(
        last_good=_unavailable_point(reason),
        first_failed=_unavailable_point(reason),
        transition=None,
        supporting_evidence=(),
        limitations=(),
        first_failed_scope=None,
    )


def _unavailable_point(reason: str) -> DiagnosticOperatorStoryPoint:
    return DiagnosticOperatorStoryPoint(
        status=DiagnosticOperatorStoryPointStatus.UNAVAILABLE,
        unavailable_reason=reason,
    )


def _sorted_positioned_events(
    positioned_events: tuple[PositionedRuntimeEvent, ...],
) -> tuple[PositionedRuntimeEvent, ...]:
    return tuple(sorted(positioned_events, key=lambda row: row.position.value))


def _failure_position_for_finding(
    finding: DiagnosticFinding,
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> ExecutionEventPosition | None:
    if finding.supporting_positions:
        return max(finding.supporting_positions, key=lambda pos: pos.value)
    if finding.failure_boundary is not None:
        event_id = finding.failure_boundary.supporting_event_id
        row = _row_for_event_id(positioned, event_id)
        if row is not None:
            return row.position
    if finding.supporting_event_ids:
        positions: list[ExecutionEventPosition] = []
        for event_id in finding.supporting_event_ids:
            row = _row_for_event_id(positioned, event_id)
            if row is not None:
                positions.append(row.position)
        if positions:
            return max(positions, key=lambda pos: pos.value)
    return None


def _select_first_failure_finding(
    findings: list[DiagnosticFinding],
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> DiagnosticFinding:
    def sort_key(finding: DiagnosticFinding) -> tuple[int, int, str]:
        position = _failure_position_for_finding(finding, positioned)
        pos_val = position.value if position is not None else 10**9
        kind_order = _FINDING_KIND_ORDER.get(finding.kind, 99)
        event_key = ""
        if finding.supporting_event_ids:
            event_key = str(finding.supporting_event_ids[0])
        return (pos_val, kind_order, event_key)

    return min(findings, key=sort_key)


def _row_at_position(
    positioned: tuple[PositionedRuntimeEvent, ...],
    position: ExecutionEventPosition,
) -> PositionedRuntimeEvent | None:
    for row in positioned:
        if row.position.value == position.value:
            return row
    return None


def _row_for_event_id(
    positioned: tuple[PositionedRuntimeEvent, ...],
    event_id: EventId,
) -> PositionedRuntimeEvent | None:
    for row in positioned:
        if row.event.event_id == event_id:
            return row
    return None


def _select_last_good_predecessor(
    positioned: tuple[PositionedRuntimeEvent, ...],
    failure_position: ExecutionEventPosition,
) -> PositionedRuntimeEvent | None:
    predecessor: PositionedRuntimeEvent | None = None
    for row in positioned:
        if row.position.value >= failure_position.value:
            break
        if row.event.event_type in _FAILURE_RUNTIME_EVENT_TYPES:
            continue
        predecessor = row
    return predecessor


def _point_from_row(
    row: PositionedRuntimeEvent | None,
    *,
    finding_kind: DiagnosticFindingKind | None,
) -> DiagnosticOperatorStoryPoint:
    if row is None:
        return _unavailable_point("anchor_event_unavailable")
    event = row.event
    return DiagnosticOperatorStoryPoint(
        status=DiagnosticOperatorStoryPointStatus.AVAILABLE,
        execution_id=event.execution_id,
        event_id=event.event_id,
        position=row.position,
        event_type_label=event.event_type.value,
        observed_at=event.timestamp,
        finding_kind=finding_kind,
    )


def _build_transition(
    last_good: DiagnosticOperatorStoryPoint,
    first_failed: DiagnosticOperatorStoryPoint,
    positioned: tuple[PositionedRuntimeEvent, ...],
) -> DiagnosticOperatorStoryTransition | None:
    if (
        last_good.position is None
        or first_failed.position is None
        or last_good.status is not DiagnosticOperatorStoryPointStatus.AVAILABLE
        or first_failed.status is not DiagnosticOperatorStoryPointStatus.AVAILABLE
    ):
        return None
    intervening = sum(
        1
        for row in positioned
        if last_good.position.value < row.position.value < first_failed.position.value
    )
    return DiagnosticOperatorStoryTransition(
        from_position=last_good.position,
        to_position=first_failed.position,
        intervening_event_count=intervening,
        is_causal_claim=False,
    )


def _evidence_refs_for_finding(
    finding: DiagnosticFinding,
) -> tuple[DiagnosticOperatorStoryEvidenceRef, ...]:
    refs: list[DiagnosticOperatorStoryEvidenceRef] = []
    position = (
        max(finding.supporting_positions, key=lambda pos: pos.value)
        if finding.supporting_positions
        else None
    )
    for event_id in finding.supporting_event_ids:
        refs.append(
            DiagnosticOperatorStoryEvidenceRef(
                execution_id=finding.execution_id,
                event_id=event_id,
                position=position,
            )
        )
    for evidence_id in finding.supporting_evidence_ids:
        refs.append(
            DiagnosticOperatorStoryEvidenceRef(
                execution_id=finding.execution_id,
                evidence_id=evidence_id,
                position=position,
            )
        )
    return tuple(refs)


def _evidence_refs_for_row(
    row: PositionedRuntimeEvent | None,
) -> tuple[DiagnosticOperatorStoryEvidenceRef, ...]:
    if row is None:
        return ()
    event = row.event
    return (
        DiagnosticOperatorStoryEvidenceRef(
            execution_id=event.execution_id,
            event_id=event.event_id,
            position=row.position,
        ),
    )


def _dedupe_evidence_refs(
    refs: tuple[DiagnosticOperatorStoryEvidenceRef, ...],
) -> tuple[DiagnosticOperatorStoryEvidenceRef, ...]:
    seen: set[tuple[object, ...]] = set()
    ordered: list[DiagnosticOperatorStoryEvidenceRef] = []
    for ref in refs:
        key = (
            str(ref.execution_id) if ref.execution_id is not None else None,
            str(ref.event_id) if ref.event_id is not None else None,
            str(ref.evidence_id) if ref.evidence_id is not None else None,
            ref.position.value if ref.position is not None else None,
        )
        if key in seen:
            continue
        seen.add(key)
        ordered.append(ref)
    return tuple(
        sorted(
            ordered,
            key=lambda ref: (
                ref.position.value if ref.position is not None else 10**9,
                str(ref.event_id or ""),
                str(ref.evidence_id or ""),
            ),
        )
    )


__all__ = [
    "DiagnosticOperatorStoryIntegrityError",
    "project_operator_story",
]
