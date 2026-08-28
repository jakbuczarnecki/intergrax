# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
from dataclasses import replace

import copy

import pytest

from intergrax.contracts.evidence_claims import (
    ClaimResolution,
    EvidenceBackedClaim,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.application.domain_reasoning import (
    IncidentEvidenceIds,
    IncidentObservations,
    ObservedComparison,
    ObservedStaffingAttendance,
    ObservedStaffingSchedule,
    ObservedTelemetry,
    ObservedThroughput,
    ObservedWorkload,
    comparison_weakens_overload,
    derive_hypothesis_dispositions,
    observations_from_evidence_nodes,
    telemetry_supports_degradation,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import (
    build_forged_h3_claim_set,
    evaluate_mutated_evidence_fails,
    evaluate_scenario_run,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import (
    TelemetryAvailability,
    build_resolved_fixture,
)
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    DIAGNOSIS_KIND,
    H2_CLAIM_ID,
    INCIDENT_EVIDENCE_IDS,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    build_runtime_bundle,
    execute_resolved_skeleton,
)
from platform_proofs.scenarios.ai_incident_investigation.application.validation import (
    COMPARISON_CONTENT_ERROR,
    H2_DISPOSITION_ERROR,
    H2_FALLBACK_ERROR,
    TELEMETRY_CONTENT_ERROR,
    validate_claim_set_against_observations,
)

pytestmark = pytest.mark.unit


def _validation_payload(
    result,
    *,
    evidence_nodes: list[dict[str, object]],
    active_hypothesis: str = "H3",
    completion_mode: str = "supported_diagnosis",
    claim_set: dict | None = None,
    extra_bindings: list[dict[str, str]] | None = None,
) -> dict[str, object]:
    bindings = list(result.claim_hypothesis_bindings)
    if extra_bindings:
        bindings = [
            item
            for item in bindings
            if item.get("hypothesis_id")
            not in {binding.get("hypothesis_id") for binding in extra_bindings}
        ] + extra_bindings
    return {
        "claim_set": claim_set or result.claim_set,
        "claim_hypothesis_bindings": bindings,
        "evidence_nodes": evidence_nodes,
        "active_hypothesis": active_hypothesis,
        "completion_mode": completion_mode,
    }


def _happy_observations() -> IncidentObservations:
    fixture = build_resolved_fixture()
    return observations_from_evidence_nodes(
        (
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.workload,
                "payload": {
                    "order_volume_delta_pct": fixture.workload_incident.order_volume_delta_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.throughput,
                "payload": {
                    "target_attainment_pct": fixture.throughput_incident.target_attainment_pct,
                    "baseline_attainment_pct": fixture.throughput_incident.baseline_attainment_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.staffing_schedule,
                "payload": {
                    "scheduled_headcount": fixture.staffing_preliminary.scheduled_headcount,
                    "required_headcount": fixture.staffing_preliminary.required_headcount,
                    "record_valid_from": fixture.staffing_preliminary.record_valid_for.observed_from.isoformat(),
                    "record_valid_to": fixture.staffing_preliminary.record_valid_for.observed_to.isoformat(),
                    "window_observed_from": fixture.staffing_preliminary.window.observed_from.isoformat(),
                    "window_observed_to": fixture.staffing_preliminary.window.observed_to.isoformat(),
                },
            },
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.staffing_attendance,
                "payload": {
                    "confirmed_headcount": fixture.staffing_attendance.confirmed_headcount,
                },
            },
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.comparison,
                "payload": {
                    "workload_delta_pct": fixture.comparison.workload_delta_pct,
                    "comparison_attainment_pct": fixture.comparison.target_attainment_pct,
                    "reference_attainment_pct": fixture.comparison.reference_attainment_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": INCIDENT_EVIDENCE_IDS.telemetry,
                "payload": {
                    "availability": "available",
                    "signal_state": fixture.telemetry.signal_state,
                    "complex_assembly_throughput_pct": fixture.telemetry.complex_assembly_throughput_pct,
                    "baseline_throughput_pct": fixture.telemetry.baseline_throughput_pct,
                    "admissible": True,
                },
            },
        ),
        INCIDENT_EVIDENCE_IDS,
    )


def test_derive_hypothesis_dispositions_has_no_fixture_leak_in_signature() -> None:
    sig = inspect.signature(derive_hypothesis_dispositions)
    param_names = set(sig.parameters)
    forbidden = {"private_truth", "expected_hypothesis", "initiating_factor_code", "is_revision"}
    assert forbidden.isdisjoint(param_names)
    for param in sig.parameters.values():
        assert "IncidentFixture" not in str(param.annotation)


def test_happy_fixture_derives_h1_superseded_h2_rejected_h3_supported() -> None:
    assessment = derive_hypothesis_dispositions(_happy_observations(), INCIDENT_EVIDENCE_IDS)
    assert assessment.h1.disposition is ClaimResolution.SUPERSEDED
    assert assessment.h2.disposition is ClaimResolution.REJECTED
    assert assessment.h3.disposition is ClaimResolution.SUPPORTED
    assert INCIDENT_EVIDENCE_IDS.comparison in assessment.h1.contradicting_evidence_ids


def test_comparison_degraded_attainment_does_not_supersede_h1() -> None:
    observations = _happy_observations()
    degraded_comparison = ObservedComparison(
        workload_delta_pct=24.0,
        comparison_attainment_pct=70.0,
        reference_attainment_pct=78.0,
        admissible=True,
    )
    observations = replace(observations, comparison=degraded_comparison)
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h1.disposition is not ClaimResolution.SUPERSEDED
    assert assessment.h3.disposition is not ClaimResolution.SUPPORTED
    assert not comparison_weakens_overload(
        observations.workload, observations.throughput, degraded_comparison
    )


def test_real_understaffing_prevents_h2_rejection() -> None:
    incident_window = build_resolved_fixture().staffing_preliminary.window
    observations = _happy_observations()
    admissible_schedule = ObservedStaffingSchedule(
        scheduled_headcount=8,
        required_headcount=12,
        record_valid_from=incident_window.observed_from.isoformat(),
        record_valid_to=incident_window.observed_to.isoformat(),
        window_observed_from=incident_window.observed_from.isoformat(),
        window_observed_to=incident_window.observed_to.isoformat(),
    )
    shortage_attendance = ObservedStaffingAttendance(confirmed_headcount=8)
    observations = replace(
        observations,
        staffing_schedule=admissible_schedule,
        staffing_attendance=shortage_attendance,
    )
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h2.disposition is not ClaimResolution.REJECTED
    assert assessment.h2.disposition is ClaimResolution.SUPPORTED


def test_healthy_telemetry_prevents_h3_support() -> None:
    observations = _happy_observations()
    healthy_telemetry = ObservedTelemetry(
        availability=TelemetryAvailability.AVAILABLE,
        signal_state="healthy",
        complex_assembly_throughput_pct=90.0,
        baseline_throughput_pct=91.0,
        admissible=True,
    )
    observations = replace(observations, telemetry=healthy_telemetry)
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h3.disposition is not ClaimResolution.SUPPORTED
    assert not telemetry_supports_degradation(healthy_telemetry)


def test_critical_mutation_same_id_different_payload_different_h3() -> None:
    degraded = ObservedTelemetry(
        availability=TelemetryAvailability.AVAILABLE,
        signal_state="intermittent_degraded",
        complex_assembly_throughput_pct=62.0,
        baseline_throughput_pct=91.0,
        admissible=True,
    )
    healthy = ObservedTelemetry(
        availability=TelemetryAvailability.AVAILABLE,
        signal_state="healthy",
        complex_assembly_throughput_pct=90.0,
        baseline_throughput_pct=91.0,
        admissible=True,
    )
    base = _happy_observations()
    degraded_assessment = derive_hypothesis_dispositions(
        replace(base, telemetry=degraded), INCIDENT_EVIDENCE_IDS
    )
    healthy_assessment = derive_hypothesis_dispositions(
        replace(base, telemetry=healthy), INCIDENT_EVIDENCE_IDS
    )
    assert degraded_assessment.h3.disposition is ClaimResolution.SUPPORTED
    assert healthy_assessment.h3.disposition is not ClaimResolution.SUPPORTED


def test_correlation_only_h1_supported_rejected_by_critic() -> None:
    forged_claim = EvidenceBackedClaim(
        claim_id=INITIAL_CLAIM_ID,
        statement="Workload overload caused degradation — H1 supported without follow-up.",
        claim_kind=DIAGNOSIS_KIND,
        supporting_evidence_ids=(WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID),
        resolution=ClaimResolution.SUPPORTED,
    )
    claim_set = EvidenceClaimSet(claims=(forged_claim,), challenges=())
    fixture = build_resolved_fixture()
    domain_payload = {
        "claim_set": claim_set.model_dump(mode="json"),
        "evidence_nodes": [
            {
                "evidence_id": str(WORKLOAD_EVIDENCE_ID),
                "payload": {
                    "order_volume_delta_pct": fixture.workload_incident.order_volume_delta_pct,
                    "admissible": True,
                },
            },
            {
                "evidence_id": str(THROUGHPUT_EVIDENCE_ID),
                "payload": {
                    "target_attainment_pct": fixture.throughput_incident.target_attainment_pct,
                    "baseline_attainment_pct": fixture.throughput_incident.baseline_attainment_pct,
                    "admissible": True,
                },
            },
        ],
        "active_hypothesis": "H1",
        "claim_hypothesis_bindings": [
            {"claim_id": str(INITIAL_CLAIM_ID), "hypothesis_id": "H1"},
        ],
    }
    result = validate_claim_set_against_observations(claim_set, domain_payload)
    assert not result.valid
    assert result.errors



@pytest.mark.asyncio
async def test_critic_rejects_forged_h3_with_healthy_telemetry_async() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED

    mutated_nodes = list(result.evidence_nodes)
    for node in mutated_nodes:
        if str(node.get("evidence_id")) == str(TELEMETRY_EVIDENCE_ID):
            payload = node.get("payload")
            if isinstance(payload, dict):
                payload["availability"] = "available"
                payload["signal_state"] = "healthy"
                payload["complex_assembly_throughput_pct"] = 90.0

    forged = build_forged_h3_claim_set(result)
    forged_h3 = next(
        claim for claim in forged.claims if claim.resolution is ClaimResolution.SUPPORTED
    )
    validation = validate_claim_set_against_observations(
        forged,
        _validation_payload(
            result,
            evidence_nodes=mutated_nodes,
            claim_set=forged.model_dump(mode="json"),
            completion_mode="supported_diagnosis",
            extra_bindings=[
                {"claim_id": str(forged_h3.claim_id), "hypothesis_id": "H3"},
            ],
        ),
    )
    assert not validation.valid
    assert TELEMETRY_CONTENT_ERROR in validation.errors


@pytest.mark.asyncio
async def test_bad_comparison_prevents_resolved_outcome() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    assert result.outcome == OUTCOME_RESOLVED

    mutated_nodes = copy.deepcopy(list(result.evidence_nodes))
    for node in mutated_nodes:
        if str(node.get("evidence_id")) == str(COMPARISON_EVIDENCE_ID):
            payload = node.get("payload")
            if isinstance(payload, dict):
                payload["comparison_attainment_pct"] = 70.0

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    validation = validate_claim_set_against_observations(
        claim_set,
        _validation_payload(result, evidence_nodes=mutated_nodes),
    )
    assert not validation.valid
    assert COMPARISON_CONTENT_ERROR in validation.errors


@pytest.mark.asyncio
async def test_real_understaffing_prevents_resolved_h2_rejection() -> None:
    observations = _happy_observations()
    incident_window = build_resolved_fixture().staffing_preliminary.window
    admissible_schedule = ObservedStaffingSchedule(
        scheduled_headcount=8,
        required_headcount=12,
        record_valid_from=incident_window.observed_from.isoformat(),
        record_valid_to=incident_window.observed_to.isoformat(),
        window_observed_from=incident_window.observed_from.isoformat(),
        window_observed_to=incident_window.observed_to.isoformat(),
    )
    observations = replace(
        observations,
        staffing_schedule=admissible_schedule,
        staffing_attendance=ObservedStaffingAttendance(confirmed_headcount=8),
    )
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h2.disposition is not ClaimResolution.REJECTED

    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    mutated_nodes = copy.deepcopy(list(result.evidence_nodes))
    for node in mutated_nodes:
        evidence_id = str(node.get("evidence_id"))
        payload = node.get("payload")
        if not isinstance(payload, dict):
            continue
        if evidence_id == str(STAFFING_ATTENDANCE_EVIDENCE_ID):
            payload["confirmed_headcount"] = 8
        if evidence_id == str(INCIDENT_EVIDENCE_IDS.staffing_schedule):
            payload["scheduled_headcount"] = 8
            payload["required_headcount"] = 12
            payload["record_valid_from"] = incident_window.observed_from.isoformat()
            payload["record_valid_to"] = incident_window.observed_to.isoformat()

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    validation = validate_claim_set_against_observations(
        claim_set,
        _validation_payload(result, evidence_nodes=mutated_nodes),
    )
    assert not validation.valid
    assert H2_DISPOSITION_ERROR in validation.errors


@pytest.mark.asyncio
async def test_evaluator_mutation_healthy_telemetry_fails() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)

    def mutate_telemetry(payload: dict[str, object]) -> None:
        payload["availability"] = "available"
        payload["signal_state"] = "healthy"
        payload["complex_assembly_throughput_pct"] = 90.0

    assert evaluate_mutated_evidence_fails(
        result,
        evidence_id=str(TELEMETRY_EVIDENCE_ID),
        payload_mutator=mutate_telemetry,
    )


@pytest.mark.asyncio
async def test_full_happy_path_still_passes_evaluator() -> None:
    bundle = build_runtime_bundle()
    result = await execute_resolved_skeleton(bundle)
    evaluation = evaluate_scenario_run(result, bundle.fixture)
    assert evaluation.passed, evaluation.failures


def test_missing_telemetry_h3_not_supported() -> None:
    observations = replace(_happy_observations(), telemetry=None)
    assessment = derive_hypothesis_dispositions(observations, INCIDENT_EVIDENCE_IDS)
    assert assessment.h3.disposition is not ClaimResolution.SUPPORTED
