# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

"""Trace timeline evidence contracts (HEP Band 2ae · EVID-TRACE-01)."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class TraceTimelineKind(str, Enum):
    CORE_CERTIFICATION = "core_certification"
    EVIDENCE_RUN = "evidence_run"
    GENERIC_TRACE = "generic_trace"


class TraceTimelineEventKind(str, Enum):
    TIMELINE_STARTED = "timeline_started"
    CERTIFICATION_STARTED = "certification_started"
    SCENARIO_STARTED = "scenario_started"
    EVIDENCE_EMITTED = "evidence_emitted"
    SCENARIO_PASSED = "scenario_passed"
    SCENARIO_FAILED = "scenario_failed"
    REPORT_WRITTEN = "report_written"
    CERTIFICATION_COMPLETED = "certification_completed"
    TIMELINE_COMPLETED = "timeline_completed"


class TraceTimelineSourceKind(str, Enum):
    CERTIFICATION_REPORT = "certification_report"
    SCENARIO_RESULT = "scenario_result"
    EVIDENCE_REF = "evidence_ref"
    RUNTIME_EVENT = "runtime_event"
    TRACE_RECORD = "trace_record"
    GENERATED = "generated"


class TraceTimelineSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class TraceTimelineArtifactKind(str, Enum):
    REPORT_JSON = "report_json"
    REPORT_MARKDOWN = "report_markdown"
    TIMELINE_JSON = "timeline_json"
    TIMELINE_MARKDOWN = "timeline_markdown"
    TRACE_RECORD = "trace_record"
    OTHER = "other"


class TraceTimelineArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TraceTimelineArtifactKind
    path: str
    description: str = ""


class TraceTimelineSourceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: TraceTimelineSourceKind
    ref: str
    description: str = ""


class TraceTimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    kind: TraceTimelineEventKind
    sequence: int
    title: str
    message: str = ""
    severity: TraceTimelineSeverity = TraceTimelineSeverity.INFO
    source_refs: list[TraceTimelineSourceRef] = Field(default_factory=list)
    artifact_refs: list[TraceTimelineArtifactRef] = Field(default_factory=list)
    scenario_id: str | None = None
    timestamp: datetime | None = None


class TraceTimeline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0.0"
    timeline_id: str
    kind: TraceTimelineKind
    title: str
    events: list[TraceTimelineEvent]
    source_report_path: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


_SCENARIO_EVENT_KINDS: frozenset[TraceTimelineEventKind] = frozenset(
    {
        TraceTimelineEventKind.SCENARIO_STARTED,
        TraceTimelineEventKind.SCENARIO_PASSED,
        TraceTimelineEventKind.SCENARIO_FAILED,
    }
)

_ERROR_SEVERITY_ALLOWED_KINDS: frozenset[TraceTimelineEventKind] = frozenset(
    {
        TraceTimelineEventKind.SCENARIO_FAILED,
        TraceTimelineEventKind.TIMELINE_COMPLETED,
    }
)

_TIMELINE_START_KINDS: frozenset[TraceTimelineEventKind] = frozenset(
    {
        TraceTimelineEventKind.TIMELINE_STARTED,
        TraceTimelineEventKind.CERTIFICATION_STARTED,
    }
)

_TIMELINE_END_KINDS: frozenset[TraceTimelineEventKind] = frozenset(
    {
        TraceTimelineEventKind.TIMELINE_COMPLETED,
        TraceTimelineEventKind.CERTIFICATION_COMPLETED,
    }
)


def generate_timeline_event_id(
    kind: TraceTimelineEventKind,
    sequence: int,
    scenario_id: str | None = None,
) -> str:
    """Return a deterministic timeline event identifier."""
    base = f"{kind.value}:{sequence}"
    if scenario_id:
        return f"{base}:{scenario_id}"
    return base


def create_trace_timeline_event(
    *,
    kind: TraceTimelineEventKind,
    sequence: int,
    title: str,
    message: str = "",
    severity: TraceTimelineSeverity = TraceTimelineSeverity.INFO,
    source_refs: list[TraceTimelineSourceRef] | None = None,
    artifact_refs: list[TraceTimelineArtifactRef] | None = None,
    scenario_id: str | None = None,
    event_id: str | None = None,
    timestamp: datetime | None = None,
) -> TraceTimelineEvent:
    """Build a timeline event with a generated ``event_id`` when omitted."""
    resolved_event_id = event_id or generate_timeline_event_id(
        kind,
        sequence,
        scenario_id,
    )
    return TraceTimelineEvent(
        event_id=resolved_event_id,
        kind=kind,
        sequence=sequence,
        title=title,
        message=message,
        severity=severity,
        source_refs=list(source_refs or []),
        artifact_refs=list(artifact_refs or []),
        scenario_id=scenario_id,
        timestamp=timestamp,
    )


def validate_trace_timeline(timeline: TraceTimeline) -> None:
    """Raise ``ValueError`` when ``timeline`` violates EVID-TRACE-01 contracts."""
    violations: list[str] = []

    if not timeline.events:
        violations.append("timeline must contain at least one event")
        raise ValueError("; ".join(violations))

    sequences = [event.sequence for event in timeline.events]
    event_ids = [event.event_id for event in timeline.events]

    if sequences[0] != 1:
        violations.append("event sequence must start at 1")

    if len(set(sequences)) != len(sequences):
        violations.append("event sequence values must be unique")

    if sequences != sorted(sequences):
        violations.append("event sequence values must be sorted ascending")

    expected_sequences = list(range(1, len(sequences) + 1))
    if sorted(sequences) != expected_sequences:
        violations.append("event sequence values must be contiguous from 1")

    if len(set(event_ids)) != len(event_ids):
        violations.append("event_id values must be unique")

    first_kind = timeline.events[0].kind
    if first_kind not in _TIMELINE_START_KINDS:
        violations.append(
            "first event must be TIMELINE_STARTED or CERTIFICATION_STARTED"
        )

    last_kind = timeline.events[-1].kind
    if last_kind not in _TIMELINE_END_KINDS:
        violations.append(
            "last event must be TIMELINE_COMPLETED or CERTIFICATION_COMPLETED"
        )

    for event in timeline.events:
        if event.kind in _SCENARIO_EVENT_KINDS and not event.scenario_id:
            violations.append(
                f"{event.kind.value} requires scenario_id (sequence={event.sequence})"
            )

        if (
            event.severity is TraceTimelineSeverity.ERROR
            and event.kind not in _ERROR_SEVERITY_ALLOWED_KINDS
        ):
            violations.append(
                f"ERROR severity is not allowed for {event.kind.value} "
                f"(sequence={event.sequence})"
            )

        for artifact_ref in event.artifact_refs:
            if not artifact_ref.path.strip():
                violations.append(
                    f"artifact path must not be empty (sequence={event.sequence})"
                )

        for source_ref in event.source_refs:
            if not source_ref.ref.strip():
                violations.append(
                    f"source ref must not be empty (sequence={event.sequence})"
                )

    if violations:
        raise ValueError("; ".join(violations))
