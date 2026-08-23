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
    IncidentFixture,
    HypothesisId,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
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


def evaluate_scenario_run(
    result: ScenarioExecutionResult,
    fixture: IncidentFixture,
) -> ScenarioEvaluationResult:
    checks: list[str] = []
    failures: list[str] = []

    if result.tool_trace_count < 2:
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
        elif challenge.resolution is not ChallengeResolution.OPEN:
            failures.append("unresolved_challenge_should_remain_open")

    if result.evaluator_loop_iterations < 1:
        failures.append("bounded_recovery_missing")
    elif result.evaluator_loop_iterations > EVALUATOR_LOOP_MAX_ITERATIONS:
        failures.append("evaluator_loop_budget_exceeded")
    else:
        checks.append("bounded_recovery_within_platform")

    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    diagnosis_claims = [c for c in claim_set.claims if c.resolution is ClaimResolution.SUPPORTED]
    if not diagnosis_claims:
        failures.append("no_supported_diagnosis_claim")
    else:
        checks.append("terminal_supported_claim_present")
        supported = diagnosis_claims[-1]
        if TELEMETRY_EVIDENCE_ID not in supported.supporting_evidence_ids:
            failures.append("supported_claim_missing_telemetry_ref")
        else:
            checks.append("supported_claim_references_telemetry_evidence")

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

    if result.revision_used_tools:
        checks.append("follow_up_via_platform_tools")
    else:
        failures.append("follow_up_not_via_tools")

    if fixture.private_truth.initiating_factor_code != "station_signal_degraded_pattern":
        failures.append("fixture_truth_integrity")
    else:
        checks.append("private_truth_consistent")

    if str(REVISED_CLAIM_ID) in result.leak_scan_blob:
        checks.append("claim_ids_present_in_observable_artifacts")

    return ScenarioEvaluationResult(
        passed=len(failures) == 0,
        checks=tuple(checks),
        failures=tuple(failures),
    )
