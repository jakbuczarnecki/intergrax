# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from intergrax.runtime.evidence.scenario_runner import run_core_certification
from intergrax.runtime.evidence.trace_timeline_adapter import (
    build_timeline_from_certification_report,
    load_core_certification_report,
)
from intergrax.runtime.evidence.trace_timeline_contracts import (
    TraceTimelineEventKind,
    TraceTimelineKind,
    validate_trace_timeline,
)
from intergrax.runtime.evidence.trace_timeline_facets import TracePolicyOutcome

pytestmark = pytest.mark.unit


def test_load_core_certification_report_reads_report_json(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    loaded = load_core_certification_report(tmp_path / "report.json")
    assert loaded.certification_run_id == report.certification_run_id
    assert loaded.passed is True


def test_load_core_certification_report_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="certification report not found"):
        load_core_certification_report(tmp_path / "report.json")


def test_build_timeline_from_certification_report_l2_passes_validation(tmp_path: Path) -> None:
    report = run_core_certification("L2", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(
        report,
        source_report_path=str(tmp_path / "report.json"),
    )
    validate_trace_timeline(timeline)
    assert timeline.kind is TraceTimelineKind.CORE_CERTIFICATION
    assert timeline.timeline_id == report.certification_run_id
    assert timeline.events[0].kind is TraceTimelineEventKind.CERTIFICATION_STARTED
    assert timeline.events[-1].kind is TraceTimelineEventKind.CERTIFICATION_COMPLETED


def test_build_timeline_includes_policy_facet_for_tool_denied_by_policy(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(report)
    evidence_events = [
        event
        for event in timeline.events
        if event.kind is TraceTimelineEventKind.EVIDENCE_EMITTED
        and event.scenario_id == "tool_denied_by_policy"
    ]
    assert len(evidence_events) == 1
    assert evidence_events[0].facets is not None
    assert evidence_events[0].facets.policy is not None
    assert evidence_events[0].facets.policy.outcome is TracePolicyOutcome.DENIED


def test_build_timeline_skips_certification_report_emitted_scenario_events(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(report)
    started_ids = {
        event.scenario_id
        for event in timeline.events
        if event.kind is TraceTimelineEventKind.SCENARIO_STARTED
    }
    assert "certification_report_emitted" not in started_ids
    report_written = next(
        event for event in timeline.events if event.kind is TraceTimelineEventKind.REPORT_WRITTEN
    )
    assert len(report_written.artifact_refs) == 2


def test_build_timeline_l1_event_count_matches_scenario_flow(tmp_path: Path) -> None:
    report = run_core_certification("L1", output_dir=tmp_path)
    timeline = build_timeline_from_certification_report(report)
    # 1 start + 3 scenarios * (start/evidence/pass) + report_written + completed = 12
    assert len(timeline.events) == 12
