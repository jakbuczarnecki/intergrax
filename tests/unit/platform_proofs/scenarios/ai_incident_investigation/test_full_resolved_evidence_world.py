# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import evaluate_scenario_run
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    TimeWindowLabel,
    build_resolved_fixture,
    staffing_record_admissible_for_incident,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
)

pytestmark = pytest.mark.unit


def test_fixture_has_four_temporal_windows() -> None:
    fixture = build_resolved_fixture()
    labels = {
        fixture.workload_baseline.window.label,
        fixture.throughput_before.window.label,
        fixture.workload_incident.window.label,
        fixture.comparison.window.label,
    }
    assert TimeWindowLabel.BASELINE in labels
    assert TimeWindowLabel.BEFORE_INCIDENT in labels
    assert TimeWindowLabel.INCIDENT in labels
    assert TimeWindowLabel.COMPARISON in labels


def test_h1_plausible_from_fixture() -> None:
    fixture = build_resolved_fixture()
    assert fixture.workload_incident.order_volume_delta_pct > 15.0
    assert (
        fixture.throughput_incident.target_attainment_pct
        < fixture.throughput_incident.baseline_attainment_pct - 10.0
    )


def test_h2_staffing_preliminary_stale_for_incident() -> None:
    fixture = build_resolved_fixture()
    assert not staffing_record_admissible_for_incident(fixture.staffing_preliminary)
    assert (
        fixture.staffing_attendance.confirmed_headcount
        >= fixture.staffing_preliminary.required_headcount
    )


def test_h3_not_obvious_from_initial_observable_set() -> None:
    fixture = build_resolved_fixture()
    assert fixture.telemetry.signal_state == "intermittent_degraded"
    assert fixture.comparison.workload_delta_pct > 20.0


@pytest.mark.asyncio
async def test_no_telemetry_in_initial_evidence_nodes() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    initial_ids = {str(node["evidence_id"]) for node in result.initial_evidence_nodes}
    assert str(TELEMETRY_EVIDENCE_ID) not in initial_ids
    assert str(COMPARISON_EVIDENCE_ID) not in initial_ids
    assert str(STAFFING_ATTENDANCE_EVIDENCE_ID) not in initial_ids
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in initial_ids


@pytest.mark.asyncio
async def test_full_resolved_evidence_world_passes_evaluator() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed, evaluation.failures


@pytest.mark.asyncio
async def test_comparison_and_staffing_follow_up_gathered() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    observable = {str(node["evidence_id"]) for node in result.evidence_nodes}
    assert str(COMPARISON_EVIDENCE_ID) in observable
    assert str(STAFFING_PRELIMINARY_EVIDENCE_ID) in observable
    assert str(STAFFING_ATTENDANCE_EVIDENCE_ID) in observable
    assert str(TELEMETRY_EVIDENCE_ID) in observable

