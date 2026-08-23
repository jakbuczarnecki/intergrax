# © Artur Czarnecki. All rights reserved.

"""Deterministic scenario evaluator — inspects observable artifacts and private truth post-run."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.evidence_claims import ClaimResolution, EvidenceClaimSet
from platform_proofs.scenarios.ai_incident_investigation.fixtures import (
    FORBIDDEN_LEAK_MARKERS,
    IncidentFixture,
    HypothesisId,
)
from platform_proofs.scenarios.ai_incident_investigation.investigator_agent import (
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.scenario import ScenarioExecutionResult


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

    if result.evaluator_loop_iterations < 1:
        failures.append("bounded_recovery_missing")
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

    leaked = _collect_leakage_strings(result)
    if leaked:
        failures.append(f"ground_truth_leak:{','.join(leaked)}")
    else:
        checks.append("ground_truth_isolated")

    if result.revision_used_tools:
        checks.append("follow_up_via_platform_tools")
    else:
        failures.append("follow_up_not_via_tools")

    # Private truth comparison — evaluator only
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
