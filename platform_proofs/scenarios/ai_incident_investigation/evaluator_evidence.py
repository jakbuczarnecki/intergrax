# © Artur Czarnecki. All rights reserved.

"""Project ScenarioEvaluationResult into canonical EvaluatorSummaryEvidence."""

from __future__ import annotations

import re

from platform_proofs.scenarios.ai_incident_investigation.evaluator import ScenarioEvaluationResult
from scripts.proof.intergrax_platform_proof_evidence import (
    EvaluatorCheckEvidence,
    EvaluatorSummaryEvidence,
    proof_authored_report_safe_text,
)

EVALUATOR_DISPLAY_NAME = "AI Incident Investigation deterministic evaluator"

_CHECK_LABELS: dict[str, tuple[str, str]] = {
    "no_initial_decisive_telemetry": (
        "No decisive telemetry before revision",
        "Initial investigation did not expose resolving telemetry prematurely.",
    ),
    "initial_observable_excludes_follow_up_evidence": (
        "Follow-up evidence gathered after challenge",
        "Comparison, attendance, and telemetry were not visible in the initial pass.",
    ),
    "h1_initially_plausible": (
        "Initial overload hypothesis was genuinely plausible",
        "Workload rose while throughput fell during the incident window.",
    ),
    "tool_runtime_exercised": (
        "Platform ToolRuntime was exercised",
        "Investigation used bounded platform tool invocations.",
    ),
    "critic_falsification_occurred": (
        "Critic issued a falsification challenge",
        "Unsupported causal inference was challenged before diagnosis acceptance.",
    ),
    "failed_critic_verdict_provenance": (
        "Failed critic verdict recorded",
        "The initial unsupported-inference rejection is auditable.",
    ),
    "critic_challenge_mapped_from_verdict": (
        "Challenge mapped from critic verdict",
        "The evidence challenge reflects the critic rejection.",
    ),
    "challenge_satisfied_after_revision": (
        "Challenge satisfied after targeted follow-up",
        "Follow-up evidence addressed the falsification challenge.",
    ),
    "satisfied_challenge_includes_telemetry_evidence": (
        "Satisfied challenge includes telemetry",
        "Resolving telemetry is part of the satisfied challenge basis.",
    ),
    "satisfied_challenge_includes_comparison_evidence": (
        "Satisfied challenge includes comparison evidence",
        "Comparison evidence is part of the satisfied challenge basis.",
    ),
    "telemetry_evidence_observable_in_graph": (
        "Telemetry observable in evidence graph",
        "Station telemetry appears in the auditable evidence graph.",
    ),
    "bounded_recovery_within_platform": (
        "Bounded recovery within platform limits",
        "Revision stayed within the configured evaluator-loop budget.",
    ),
    "follow_up_via_platform_tools": (
        "Follow-up used platform tools",
        "Targeted evidence was gathered through ToolRuntime.",
    ),
    "comparison_evidence_gathered": (
        "Comparison evidence gathered",
        "A high-load comparison window was queried for context.",
    ),
    "staffing_preliminary_gathered": (
        "Preliminary staffing evidence gathered",
        "Schedule export for the incident window was collected.",
    ),
    "staffing_attendance_gathered": (
        "Attendance evidence gathered",
        "Confirmed attendance for the incident window was collected.",
    ),
    "staffing_preliminary_stale_for_incident": (
        "Preliminary staffing treated as stale",
        "Stale roster export was not treated as decisive support.",
    ),
    "h2_rejected_with_evidence": (
        "Understaffing hypothesis rejected",
        "Attendance evidence removed understaffing as initiating cause.",
    ),
    "stale_staffing_not_final_support": (
        "Stale staffing not promoted to support",
        "Preliminary roster alone did not support a diagnosis.",
    ),
    "comparison_weakens_h1_from_runtime": (
        "Comparison evidence weakened H1",
        "High-load comparison weakened overload-only explanation.",
    ),
    "telemetry_supports_h3_from_runtime": (
        "Telemetry supported H3",
        "Station telemetry supported equipment/process degradation.",
    ),
    "terminal_supported_claim_present": (
        "Terminal supported diagnosis present",
        "A bounded supported diagnosis remained at completion.",
    ),
    "final_supported_claim_is_h3": (
        "Final supported claim is H3",
        "Equipment/process degradation was the accepted diagnosis.",
    ),
    "supported_claim_references_telemetry_evidence": (
        "Supported claim cites telemetry",
        "Accepted diagnosis references station telemetry evidence.",
    ),
    "supported_claim_references_comparison_evidence": (
        "Supported claim cites comparison evidence",
        "Accepted diagnosis references comparison evidence.",
    ),
    "h1_materially_weakened": (
        "H1 materially weakened",
        "Overload hypothesis was superseded rather than supported.",
    ),
    "all_cited_evidence_in_graph": (
        "All cited evidence present in graph",
        "Every cited evidence identifier appears in the evidence graph.",
    ),
    "critic_content_validation_passed": (
        "Critic content validation passed",
        "Claim and challenge content passed structural validation.",
    ),
    "resolved_outcome": (
        "Incident outcome is RESOLVED",
        "Execution completed with a bounded supported diagnosis.",
    ),
    "ground_truth_isolated": (
        "Private oracle isolated from report surface",
        "Fixture oracle data did not leak into observable artifacts.",
    ),
    "private_truth_consistent": (
        "Evaluator oracle consistency check passed",
        "Post-run oracle checks matched the resolved execution path.",
    ),
    "private_truth_consistent_despite_unresolved": (
        "Evaluator oracle consistency check passed",
        "Post-run oracle checks matched the unresolved execution path.",
    ),
    "final_summary_bounded": (
        "Final summary stayed bounded",
        "Terminal narrative did not over-claim beyond evidence.",
    ),
    "challenge_remains_open_without_resolving_evidence": (
        "Challenge remains open",
        "Falsification challenge stayed open without resolving telemetry.",
    ),
    "telemetry_evidence_gathered": (
        "Telemetry source was queried",
        "Station telemetry was requested for the incident window.",
    ),
    "telemetry_unavailable_observed": (
        "Telemetry unavailable for incident window",
        "Telemetry source responded without an admissible observation.",
    ),
    "no_fabricated_telemetry_measurements": (
        "No measurement was fabricated",
        "Unavailable telemetry did not produce invented readings.",
    ),
    "h1_final_not_supported": (
        "H1 not supported at completion",
        "Overload was not accepted as root cause.",
    ),
    "h3_insufficient_evidence": (
        "H3 insufficient evidence",
        "Equipment/process degradation could not be supported.",
    ),
    "zero_supported_diagnosis": (
        "No supported root-cause diagnosis",
        "No claim reached supported diagnosis at completion.",
    ),
    "unresolved_outcome": (
        "Incident outcome is UNRESOLVED",
        "Execution completed without an accepted root-cause diagnosis.",
    ),
    "final_critic_accepted_unresolved_completion": (
        "Final Critic accepted bounded uncertainty",
        "Completion remained epistemically unresolved rather than forced.",
    ),
}


def _humanize_check_id(check_id: str) -> str:
    normalized = check_id.replace("_missing", " missing").replace("_", " ").strip()
    return normalized[:1].upper() + normalized[1:]


def _label_for_check(check_id: str) -> tuple[str, str]:
    if check_id in _CHECK_LABELS:
        return _CHECK_LABELS[check_id]
    return (
        _humanize_check_id(check_id),
        f"Deterministic evaluator check: {_humanize_check_id(check_id)}.",
    )


def project_scenario_evaluation_to_evidence(
    evaluation: ScenarioEvaluationResult,
) -> EvaluatorSummaryEvidence:
    """Map runtime evaluation checks into canonical v3 evaluator evidence."""
    checks: list[EvaluatorCheckEvidence] = []
    for check_id in evaluation.checks:
        label, explanation = _label_for_check(check_id)
        checks.append(
            EvaluatorCheckEvidence(
                check_id=check_id,
                label=label,
                passed=True,
                explanation=proof_authored_report_safe_text(explanation),
            )
        )
    for failure_id in evaluation.failures:
        label, explanation = _label_for_check(failure_id)
        checks.append(
            EvaluatorCheckEvidence(
                check_id=failure_id,
                label=label,
                passed=False,
                explanation=proof_authored_report_safe_text(explanation),
            )
        )
    return EvaluatorSummaryEvidence(
        passed=evaluation.passed,
        checks=tuple(checks),
        failure_reasons=tuple(evaluation.failures),
    )


_REPRESENTATIVE_CHECK_IDS: dict[str, tuple[str, ...]] = {
    "resolved": (
        "h1_initially_plausible",
        "critic_falsification_occurred",
        "follow_up_via_platform_tools",
        "comparison_weakens_h1_from_runtime",
        "telemetry_supports_h3_from_runtime",
        "resolved_outcome",
    ),
    "unresolved": (
        "h1_initially_plausible",
        "critic_falsification_occurred",
        "telemetry_unavailable_observed",
        "no_fabricated_telemetry_measurements",
        "zero_supported_diagnosis",
        "final_critic_accepted_unresolved_completion",
    ),
}


def representative_check_labels(
    evaluation: ScenarioEvaluationResult,
    *,
    path_key: str,
) -> tuple[str, ...]:
    """Return reader-facing labels for top-of-report evaluator highlights."""
    by_id = {check.check_id: check.label for check in project_scenario_evaluation_to_evidence(evaluation).checks}
    labels: list[str] = []
    for check_id in _REPRESENTATIVE_CHECK_IDS[path_key]:
        label = by_id.get(check_id)
        if label is not None:
            labels.append(label)
    if not labels:
        return tuple(_label_for_check(check_id)[0] for check_id in evaluation.checks[:6])
    return tuple(labels)


def evaluator_pass_summary(evaluation: ScenarioEvaluationResult) -> tuple[int, int, bool]:
    """Return (passed_count, total_count, overall_passed) from evaluation."""
    passed_count = len(evaluation.checks)
    total_count = passed_count + len(evaluation.failures)
    return passed_count, total_count, evaluation.passed


def is_private_truth_check_id(check_id: str) -> bool:
    return bool(re.fullmatch(r"private_truth_consistent(_despite_unresolved)?", check_id))
