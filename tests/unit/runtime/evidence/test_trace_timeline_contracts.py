# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.trace_timeline_contracts import (
    TraceTimeline,
    TraceTimelineArtifactKind,
    TraceTimelineArtifactRef,
    TraceTimelineEventKind,
    TraceTimelineKind,
    TraceTimelineSeverity,
    TraceTimelineSourceKind,
    TraceTimelineSourceRef,
    create_trace_timeline_event,
    generate_timeline_event_id,
    validate_trace_timeline,
)

pytestmark = pytest.mark.unit

_EVIDENCE_ROOT = Path(__file__).resolve().parents[3] / "intergrax" / "runtime" / "evidence"


def test_create_trace_timeline_event_generates_event_id() -> None:
    event = create_trace_timeline_event(
        kind=TraceTimelineEventKind.TIMELINE_STARTED,
        sequence=1,
        title="Timeline started",
    )
    assert event.event_id == "timeline_started:1"
    assert event.kind is TraceTimelineEventKind.TIMELINE_STARTED
    assert event.sequence == 1


def test_generate_timeline_event_id_is_deterministic() -> None:
    first = generate_timeline_event_id(
        TraceTimelineEventKind.SCENARIO_PASSED,
        4,
        "basic_run_completed",
    )
    second = generate_timeline_event_id(
        TraceTimelineEventKind.SCENARIO_PASSED,
        4,
        "basic_run_completed",
    )
    assert first == second == "scenario_passed:4:basic_run_completed"


def test_validate_trace_timeline_minimal_passes() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-minimal",
        kind=TraceTimelineKind.GENERIC_TRACE,
        title="Minimal timeline",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_STARTED,
                sequence=1,
                title="Started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_COMPLETED,
                sequence=2,
                title="Completed",
            ),
        ],
    )
    validate_trace_timeline(timeline)


def test_validate_trace_timeline_certification_passes() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-cert",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Core certification timeline",
        source_report_path="build/evidence/core_certification/report.json",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_STARTED,
                sequence=2,
                title="Scenario started",
                scenario_id="basic_run_completed",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.EVIDENCE_EMITTED,
                sequence=3,
                title="Evidence emitted",
                source_refs=[
                    TraceTimelineSourceRef(
                        kind=TraceTimelineSourceKind.EVIDENCE_REF,
                        ref="mock:basic_run_completed",
                    )
                ],
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_PASSED,
                sequence=4,
                title="Scenario passed",
                scenario_id="basic_run_completed",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.REPORT_WRITTEN,
                sequence=5,
                title="Report written",
                artifact_refs=[
                    TraceTimelineArtifactRef(
                        kind=TraceTimelineArtifactKind.REPORT_JSON,
                        path="build/evidence/core_certification/report.json",
                    )
                ],
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=6,
                title="Certification completed",
            ),
        ],
    )
    validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_no_events() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-empty",
        kind=TraceTimelineKind.GENERIC_TRACE,
        title="Empty timeline",
        events=[],
    )
    with pytest.raises(ValueError, match="at least one event"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_sequence_not_starting_at_one() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-seq",
        kind=TraceTimelineKind.GENERIC_TRACE,
        title="Bad sequence start",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_STARTED,
                sequence=2,
                title="Started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_COMPLETED,
                sequence=3,
                title="Completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="must start at 1"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_duplicate_sequence() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-dup-seq",
        kind=TraceTimelineKind.GENERIC_TRACE,
        title="Duplicate sequence",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_STARTED,
                sequence=1,
                title="Started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_COMPLETED,
                sequence=1,
                title="Completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="sequence values must be unique"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_duplicate_event_id() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-dup-id",
        kind=TraceTimelineKind.GENERIC_TRACE,
        title="Duplicate event id",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_STARTED,
                sequence=1,
                title="Started",
                event_id="duplicate-id",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.TIMELINE_COMPLETED,
                sequence=2,
                title="Completed",
                event_id="duplicate-id",
            ),
        ],
    )
    with pytest.raises(ValueError, match="event_id values must be unique"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_scenario_passed_without_scenario_id() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-no-scenario",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Missing scenario id",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_PASSED,
                sequence=2,
                title="Scenario passed",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=3,
                title="Certification completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="scenario_passed requires scenario_id"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_empty_artifact_path() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-artifact",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Empty artifact path",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.REPORT_WRITTEN,
                sequence=2,
                title="Report written",
                artifact_refs=[
                    TraceTimelineArtifactRef(
                        kind=TraceTimelineArtifactKind.REPORT_JSON,
                        path="   ",
                    )
                ],
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=3,
                title="Certification completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="artifact path must not be empty"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_empty_source_ref() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-source",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Empty source ref",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.EVIDENCE_EMITTED,
                sequence=2,
                title="Evidence emitted",
                source_refs=[
                    TraceTimelineSourceRef(
                        kind=TraceTimelineSourceKind.EVIDENCE_REF,
                        ref="",
                    )
                ],
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=3,
                title="Certification completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="source ref must not be empty"):
        validate_trace_timeline(timeline)


def test_validate_trace_timeline_rejects_error_severity_for_scenario_passed() -> None:
    timeline = TraceTimeline(
        timeline_id="timeline-severity",
        kind=TraceTimelineKind.CORE_CERTIFICATION,
        title="Invalid severity",
        events=[
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_STARTED,
                sequence=1,
                title="Certification started",
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.SCENARIO_PASSED,
                sequence=2,
                title="Scenario passed",
                scenario_id="basic_run_completed",
                severity=TraceTimelineSeverity.ERROR,
            ),
            create_trace_timeline_event(
                kind=TraceTimelineEventKind.CERTIFICATION_COMPLETED,
                sequence=3,
                title="Certification completed",
            ),
        ],
    )
    with pytest.raises(ValueError, match="ERROR severity is not allowed for scenario_passed"):
        validate_trace_timeline(timeline)


def test_evidence_modules_have_no_applications_or_agents_imports() -> None:
    forbidden = ("applications.", "agents.", "from applications", "from agents")
    for path in _EVIDENCE_ROOT.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} contains forbidden import token: {token}"
