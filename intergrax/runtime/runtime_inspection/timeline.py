# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded deterministic timeline projection from execution reconstruction."""

from __future__ import annotations

from intergrax.contracts.execution_reconstruction import ExecutionReconstruction
from intergrax.contracts.execution_reconstruction_models import RuntimeHistoryCompleteness
from intergrax.contracts.runtime_inspection.completeness import RuntimeInspectionCompleteness
from intergrax.contracts.runtime_inspection.sections import (
    RuntimeInspectionTimelineDomain,
    RuntimeInspectionTimelineEntry,
    RuntimeInspectionTimelineSection,
)
from intergrax.runtime.runtime_inspection.redaction import sanitize_inspection_text


def build_timeline_section(
    reconstruction: ExecutionReconstruction,
    *,
    source_id: str,
    limit: int,
) -> RuntimeInspectionTimelineSection:
    positioned = tuple(
        sorted(
            reconstruction.positioned_events,
            key=lambda item: (item.position.value, str(item.event.event_id)),
        ),
    )
    causal = tuple(
        sorted(
            reconstruction.causal_evidence,
            key=lambda item: (item.recorded_at, str(item.evidence_id)),
        ),
    )
    entries: list[RuntimeInspectionTimelineEntry] = []
    for positioned_event in positioned:
        event = positioned_event.event
        summary = sanitize_inspection_text(
            f"{event.event_type.value}:{event.event_kind or event.phase.value}",
        )
        entries.append(
            RuntimeInspectionTimelineEntry(
                observed_at=event.timestamp,
                sequence_key=positioned_event.position.value,
                domain=RuntimeInspectionTimelineDomain.RUNTIME_EVENT,
                kind=event.event_type.value,
                safe_summary=summary,
                event_id=event.event_id,
            ),
        )
    for evidence in causal:
        summary = sanitize_inspection_text(evidence.relation_kind.value)
        entries.append(
            RuntimeInspectionTimelineEntry(
                observed_at=evidence.recorded_at,
                sequence_key=len(entries) + 1,
                domain=RuntimeInspectionTimelineDomain.CAUSAL_EVIDENCE,
                kind=evidence.relation_kind.value,
                safe_summary=summary,
                event_id=evidence.evidence_id,
            ),
        )
    ordered = tuple(
        sorted(
            entries,
            key=lambda item: (item.observed_at, item.sequence_key, item.domain.value, item.kind),
        ),
    )
    truncated = len(ordered) > limit
    bounded = ordered[:limit]
    completeness = (
        RuntimeInspectionCompleteness.COMPLETE
        if not truncated
        and reconstruction.runtime_history_completeness
        is RuntimeHistoryCompleteness.COMPLETE
        else RuntimeInspectionCompleteness.PARTIAL
    )
    if not bounded:
        completeness = RuntimeInspectionCompleteness.UNAVAILABLE
    return RuntimeInspectionTimelineSection(
        entries=bounded,
        is_truncated=truncated,
        completeness=completeness,
        source_id=source_id,
    )


__all__ = ["build_timeline_section"]
