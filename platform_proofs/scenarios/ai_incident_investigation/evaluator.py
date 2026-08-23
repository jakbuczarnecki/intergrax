# © Artur Czarnecki. All rights reserved.

"""Deterministic scenario evaluator — inspects observable artifacts and private truth post-run."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.evidence_claims import (
    ChallengeDefectFamily,
    ChallengeResolution,
    ClaimResolution,
    EvidenceClaimSet,
)
from platform_proofs.scenarios.ai_incident_investigation.critic_adapter import (
    UNSUPPORTED_INFERENCE_DEFECT,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures import (
    FORBIDDEN_LEAK_MARKERS,
    HypothesisId,
    IncidentFixture,
    staffing_record_admissible_for_incident,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    COMPARISON_EVIDENCE_ID,
    H2_CLAIM_ID,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    STAFFING_ATTENDANCE_EVIDENCE_ID,
    STAFFING_PRELIMINARY_EVIDENCE_ID,
    TELEMETRY_EVIDENCE_ID,
    THROUGHPUT_EVIDENCE_ID,
    WORKLOAD_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import (
    EVALUATOR_LOOP_MAX_ITERATIONS,
    ScenarioExecutionResult,
)
from platform_proofs.scenarios.ai_incident_investigation.validation import (
    UNSUPPORTED_INFERENCE_ERROR,
)


@dataclass(frozen=True, slots=True)
class ScenarioEvaluationResult:
    passed: bool
    checks: tuple[str, ...]
    failures: tuple[str, ...]


def _collect_leakage_strings(result: ScenarioExecutionResult) -> list[str]:
    leaked: list[str] = []
    for marker in FORBIDDEN_LEAK_MARKERS:
        blob = result.leak_scan_blob.lower()
        if marker.lower() in blob:
            leaked.append(marker)
    return leaked


def _observable_evidence_ids(result: ScenarioExecutionResult) -> frozenset[str]:
    return frozenset(str(node.get("evidence_id")) for node in result.evidence_nodes)


def _initial_h1_plausible(fixture: IncidentFixture) -> bool:
    return (
        fixture.workload_incident.order_volume_delta_pct > 15.0
        and fixture.throughput_incident.target_attainment_pct
        < fixture.throughput_incident.baseline_attainment_pct - 10.0
    )


def evaluate_scenario_run(
    result: ScenarioExecutionResult,
    fixture: IncidentFixture,
) -> ScenarioEvaluationResult:
    checks: list[str] = []
    failures: list[str] = []

    observable_ids = _observable_evidence_ids(result)

    if TELEMETRY_EVIDENCE_ID in observable_ids and not result.revision_pass:
        failures.append("telemetry_visible_before_revision")
    else:
        checks.append("no_initial_decisive_telemetry")

    initial_observable = frozenset(
        str(node.get("evidence_id")) for node in result.initial_evidence_nodes
    )
    if str(TELEMETRY_EVIDENCE_ID) in initial_observable:
        failures.append("telemetry_in_initial_observable_set")
    elif str(COMPARISON_EVIDENCE_ID) in initial_observable:
        failures.append("comparison_in_initial_observable_set")
    else:
        checks.append("initial_observable_excludes_follow_up_evidence")

    if not _initial_h1_plausible(fixture):
        failures.append("h1_not_plausible_from_fixture")
    else:
        checks.append("h1_initially_plausible")

    if result.tool_trace_count < 3:
        failures.append("tool_runtime_not_exercised")
    else:
        checks.append("tool_runtime_exercised")

    if not result.critic_challenged:
        failures.append("critic_falsification_missing")
    else:
        checks.append("critic_falsification_occurred")

    if result.failed_critic_verdict is None:
        failures.append("failed_critic_verdict_missing")
    elif UNSUPPORTED_INFERENCE_ERROR not in result.failed_critic_verdict.failure_reasons:
        failures.append("failed_critic_verdict_reason_mismatch")
    else:
        checks.append("failed_critic_verdict_provenance")

    if result.evidence_challenge is None:
        failures.append("evidence_challenge_missing")
    else:
        challenge = result.evidence_challenge
        if challenge.claim_id != INITIAL_CLAIM_ID:
            failures.append("challenge_target_claim_mismatch")
        elif challenge.defect_family is not ChallengeDefectFamily.UNSUPPORTED_INFERENCE:
            failures.append("challenge_defect_family_mismatch")
        elif challenge.defect_code != UNSUPPORTED_INFERENCE_DEFECT:
            failures.append("challenge_defect_code_mismatch")
        else:
            checks.append("critic_challenge_mapped_from_verdict")

        if result.critic_verdict_passed:
            if challenge.resolution is not ChallengeResolution.SATISFIED:
                failures.append("challenge_not_satisfied_after_resolution")
            else:
                checks.append("challenge_satisfied_after_revision")
                if TELEMETRY_EVIDENCE_ID not in challenge.evidence_ids:
                    failures.append("satisfied_challenge_missing_resolving_evidence")
                else:
                    checks.append("satisfied_challenge_includes_telemetry_evidence")
                if COMPARISON_EVIDENCE_ID not in challenge.evidence_ids:
                    failures.append("satisfied_challenge_missing_comparison_evidence")
                else:
                    checks.append("satisfied_challenge_includes_comparison_evidence")
                for evidence_id in (WORKLOAD_EVIDENCE_ID, THROUGHPUT_EVIDENCE_ID):
                    if evidence_id not in challenge.evidence_ids:
                        failures.append("satisfied_challenge_missing_initial_evidence")
                if str(TELEMETRY_EVIDENCE_ID) not in observable_ids:
                    failures.append("telemetry_evidence_not_in_graph")
                else:
                    checks.append("telemetry_evidence_observable_in_graph")
        elif challenge.resolution is not ChallengeResolution.OPEN:
            failures.append("unresolved_challenge_should_remain_open")
        elif TELEMETRY_EVIDENCE_ID in challenge.evidence_ids:
            failures.append("open_challenge_must_not_include_resolving_evidence")

    if result.evaluator_loop_iterations < 1:
        failures.append("bounded_recovery_missing")
    elif result.evaluator_loop_iterations > EVALUATOR_LOOP_MAX_ITERATIONS:
        failures.append("evaluator_loop_budget_exceeded")
    else:
        checks.append("bounded_recovery_within_platform")

    if not result.revision_used_tools:
        failures.append("follow_up_not_via_tools")
    else:
        checks.append("follow_up_via_platform_tools")

    if COMPARISON_EVIDENCE_ID not in observable_ids:
        failures.append("comparison_evidence_not_gathered")
    else:
        checks.append("comparison_evidence_gathered")

    if STAFFING_PRELIMINARY_EVIDENCE_ID not in observable_ids:
        failures.append("staffing_preliminary_not_gathered")
    else:
        checks.append("staffing_preliminary_gathered")

    if STAFFING_ATTENDANCE_EVIDENCE_ID not in observable_ids:
        failures.append("staffing_attendance_not_gathered")
    else:
        checks.append("staffing_attendance_gathered")

    if staffing_record_admissible_for_incident(fixture.staffing_preliminary):
        failures.append("staffing_preliminary_should_be_stale_for_incident")
    else:
        checks.append("staffing_preliminary_stale_for_incident")

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    h2_claims = [c for c in claim_set.claims if c.claim_id == H2_CLAIM_ID]
    if not h2_claims:
        failures.append("h2_claim_missing")
    elif h2_claims[0].resolution is not ClaimResolution.REJECTED:
        failures.append("h2_not_rejected")
    else:
        checks.append("h2_rejected_with_evidence")
        if STAFFING_ATTENDANCE_EVIDENCE_ID not in h2_claims[0].supporting_evidence_ids:
            failures.append("h2_rejection_missing_attendance_ref")

    for claim in claim_set.claims:
        if claim.resolution is ClaimResolution.SUPPORTED:
            for evidence_id in claim.supporting_evidence_ids:
                if str(evidence_id).startswith("evidence.staffing.schedule."):
                    failures.append("stale_staffing_used_as_final_support")
                    break
    else:
        checks.append("stale_staffing_not_final_support")

    diagnosis_claims = [c for c in claim_set.claims if c.resolution is ClaimResolution.SUPPORTED]
    if not diagnosis_claims:
        failures.append("no_supported_diagnosis_claim")
    else:
        checks.append("terminal_supported_claim_present")
        supported = diagnosis_claims[-1]
        if supported.claim_id != REVISED_CLAIM_ID:
            failures.append("final_supported_claim_not_h3")
        else:
            checks.append("final_supported_claim_is_h3")
        if TELEMETRY_EVIDENCE_ID not in supported.supporting_evidence_ids:
            failures.append("supported_claim_missing_telemetry_ref")
        else:
            checks.append("supported_claim_references_telemetry_evidence")
        if COMPARISON_EVIDENCE_ID not in supported.supporting_evidence_ids:
            failures.append("supported_claim_missing_comparison_ref")
        else:
            checks.append("supported_claim_references_comparison_evidence")

    h1_claims = [c for c in claim_set.claims if c.claim_id == INITIAL_CLAIM_ID]
    if h1_claims and h1_claims[0].resolution is ClaimResolution.PENDING:
        failures.append("h1_still_pending_at_end")
    elif h1_claims and h1_claims[0].resolution not in {
        ClaimResolution.SUPERSEDED,
        ClaimResolution.REJECTED,
    }:
        failures.append("h1_not_weakened")
    else:
        checks.append("h1_materially_weakened")

    for claim in claim_set.claims:
        for evidence_id in claim.supporting_evidence_ids:
            if str(evidence_id) not in observable_ids:
                failures.append(f"cited_evidence_not_in_graph:{evidence_id}")
    if not any(f.startswith("cited_evidence_not_in_graph:") for f in failures):
        checks.append("all_cited_evidence_in_graph")

    if result.outcome != "RESOLVED":
        failures.append(f"unexpected_outcome:{result.outcome}")
    else:
        checks.append("resolved_outcome")

    if not result.critic_verdict_passed:
        failures.append("final_critic_verdict_not_passed")

    leaked = _collect_leakage_strings(result)
    if leaked:
        failures.append(f"ground_truth_leak:{','.join(leaked)}")
    else:
        checks.append("ground_truth_isolated")

    if fixture.private_truth.expected_hypothesis is not HypothesisId.H3:
        failures.append("fixture_truth_integrity")
    else:
        checks.append("private_truth_consistent")

    if "bounded H3" not in result.terminal_summary and "equipment" not in result.terminal_summary.lower():
        failures.append("final_summary_not_bounded")
    else:
        checks.append("final_summary_bounded")

    return ScenarioEvaluationResult(
        passed=len(failures) == 0,
        checks=tuple(checks),
        failures=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class SkepticStrategyResult:
    strategy: str
    passed: bool
    reason: str


def evaluate_correlation_only_strategy(fixture: IncidentFixture) -> SkepticStrategyResult:
    """Accept H1 because workload↑ and throughput↓ — must fail."""
    plausible = _initial_h1_plausible(fixture)
    has_comparison = fixture.comparison.workload_delta_pct > 20.0
    has_telemetry = fixture.telemetry.signal_state == "intermittent_degraded"
    if plausible and not has_comparison and not has_telemetry:
        return SkepticStrategyResult("correlation_only", True, "would incorrectly pass")
    return SkepticStrategyResult(
        "correlation_only",
        False,
        "correlation_only_insufficient_without_comparison_and_telemetry",
    )


def evaluate_stale_staffing_strategy(fixture: IncidentFixture) -> SkepticStrategyResult:
    """Treat preliminary staffing shortage as current — must fail."""
    if staffing_record_admissible_for_incident(fixture.staffing_preliminary):
        return SkepticStrategyResult("stale_staffing", True, "would incorrectly pass")
    return SkepticStrategyResult(
        "stale_staffing",
        False,
        "preliminary_roster_not_admissible_for_incident_window",
    )


def evaluate_h3_without_telemetry_strategy() -> SkepticStrategyResult:
    """Select H3 without follow-up telemetry — must fail scenario acceptance."""
    return SkepticStrategyResult(
        "h3_without_telemetry",
        False,
        "telemetry_not_in_initial_observable_set",
    )
