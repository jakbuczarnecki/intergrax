# © Artur Czarnecki. All rights reserved.

"""Scenario #1 specialized HTML report sections — evidence-first comprehension layer."""

from __future__ import annotations

from intergrax.contracts.evidence_claims import ChallengeResolution, ClaimResolution
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import ScenarioEvaluationResult
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator_evidence import (
    EVALUATOR_DISPLAY_NAME,
    evaluator_pass_summary,
    representative_check_labels,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    ScenarioExecutionResult,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario_contract import (
    H2_CLAIM_ID,
    H3_CLAIM_ID,
    INITIAL_CLAIM_ID,
    REVISED_CLAIM_ID,
    TELEMETRY_EVIDENCE_ID,
)
from scripts.proof.intergrax_platform_proof_evidence import PlatformProofEvidence
from scripts.proof.intergrax_platform_proof_html_renderer import (
    RenderedReportSection,
    escape_report_html,
    render_execution_status_badge,
    render_report_safe_text,
)

_HYPOTHESIS_CLAIMS: tuple[tuple[str, str, str], ...] = (
    (str(INITIAL_CLAIM_ID), "H1", "Production overload"),
    (str(H2_CLAIM_ID), "H2", "Understaffing"),
)


def _h3_resolution(dispositions: dict[str, ClaimResolution]) -> ClaimResolution:
    if str(REVISED_CLAIM_ID) in dispositions:
        return dispositions[str(REVISED_CLAIM_ID)]
    if str(H3_CLAIM_ID) in dispositions:
        return dispositions[str(H3_CLAIM_ID)]
    return ClaimResolution.PENDING

_RESOLUTION_LABELS: dict[ClaimResolution, str] = {
    ClaimResolution.SUPERSEDED: "SUPERSEDED",
    ClaimResolution.REJECTED: "REJECTED",
    ClaimResolution.SUPPORTED: "SUPPORTED",
    ClaimResolution.INSUFFICIENT_EVIDENCE: "INSUFFICIENT EVIDENCE",
    ClaimResolution.PENDING: "PENDING",
}

_INITIAL_HYPOTHESIS = (
    "H1 — Production overload looked plausible because workload increased while throughput fell."
)
_CHALLENGE_SUMMARY = (
    "The verifier rejected correlation as sufficient causal evidence "
    "(unsupported inference challenge)."
)
_FOLLOW_UP_ITEMS = (
    "Comparison line evidence",
    "Confirmed attendance",
    "Station telemetry",
)


def incident_report_extra_css() -> str:
    return """
.incident-outcome-grid {
  display: grid;
  gap: 0.75rem;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  margin: 1rem 0;
}
.incident-outcome-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.85rem 1rem;
  background: var(--surface);
}
.incident-journey {
  display: grid;
  gap: 0.5rem;
  margin: 1rem 0;
}
.incident-journey-step {
  border-left: 3px solid var(--accent);
  padding: 0.5rem 0 0.5rem 0.75rem;
  background: #f8fafc;
  border-radius: 0 6px 6px 0;
}
.incident-hypothesis-grid {
  display: grid;
  gap: 0.6rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin: 0.75rem 0;
}
.incident-hypothesis-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.75rem;
  background: var(--surface);
}
.incident-hypothesis-card .disposition {
  font-weight: 700;
  margin-top: 0.35rem;
}
.incident-pass-unresolved-note {
  border: 1px solid #fde68a;
  background: #fffbeb;
  border-radius: 8px;
  padding: 0.85rem 1rem;
  margin: 0.75rem 0 1rem;
}
.incident-evaluator-summary {
  margin-top: 0.75rem;
}
""".strip()


def _claim_dispositions(evidence: PlatformProofEvidence) -> dict[str, ClaimResolution]:
    by_id: dict[str, ClaimResolution] = {}
    for claim in evidence.evidence_claims.claims:
        by_id[claim.claim_id] = claim.resolution
    return by_id


def _challenge_resolution(evidence: PlatformProofEvidence) -> ChallengeResolution | None:
    if not evidence.evidence_claims.challenges:
        return None
    return evidence.evidence_claims.challenges[0].resolution


def _telemetry_unavailable(evidence: PlatformProofEvidence) -> bool:
    for node in evidence.evidence_graph.nodes:
        if node.evidence_id == str(TELEMETRY_EVIDENCE_ID):
            summary = render_report_safe_text(node.summary).lower()
            return "unavailable" in summary
    return False


def _outcome_badge(outcome: str) -> str:
    css = "status-pass" if outcome == OUTCOME_RESOLVED else "status-blocked"
    return (
        f'<span class="status-badge {css}" role="status" '
        f'aria-label="{escape_report_html(outcome)}">'
        f'<span class="status-label">{escape_report_html(outcome)}</span></span>'
    )


def _hypothesis_cards(dispositions: dict[str, ClaimResolution]) -> str:
    cards: list[str] = []
    for claim_id, code, title in _HYPOTHESIS_CLAIMS:
        resolution = dispositions.get(claim_id, ClaimResolution.PENDING)
        label = _RESOLUTION_LABELS.get(resolution, resolution.value.upper())
        cards.append(
            '<article class="incident-hypothesis-card">'
            f"<p><strong>{escape_report_html(code)}</strong> — {escape_report_html(title)}</p>"
            f'<p class="disposition">{escape_report_html(label)}</p>'
            "</article>"
        )
    h3_resolution = _h3_resolution(dispositions)
    h3_label = _RESOLUTION_LABELS.get(h3_resolution, h3_resolution.value.upper())
    cards.append(
        '<article class="incident-hypothesis-card">'
        "<p><strong>H3</strong> — Equipment/process degradation</p>"
        f'<p class="disposition">{escape_report_html(h3_label)}</p>'
        "</article>"
    )
    return '<div class="incident-hypothesis-grid">' + "".join(cards) + "</div>"


def _journey_html(*, resolved_path: bool, challenge_open: bool) -> str:
    steps = [
        "Initial H1 plausible — workload rose while throughput fell.",
        "Critic challenge — correlation alone is not causation.",
        "Targeted follow-up via platform tools.",
        "Comparison, attendance, and telemetry gathered.",
    ]
    if resolved_path:
        steps.append("Final dispositions — H1 superseded, H2 rejected, H3 supported.")
    else:
        steps.append(
            "Final dispositions — H1 superseded, H2 rejected, H3 insufficient evidence."
        )
        if challenge_open:
            steps[-1] += " Challenge remains OPEN."
    return (
        '<div class="incident-journey">'
        + "".join(f'<div class="incident-journey-step"><p>{escape_report_html(s)}</p></div>' for s in steps)
        + "</div>"
    )


def _why_resolved_or_unresolved(
    *,
    outcome: str,
    evidence: PlatformProofEvidence,
    challenge_open: bool,
) -> str:
    if outcome == OUTCOME_RESOLVED:
        return (
            "Telemetry showed intermittent station degradation and comparison evidence "
            "weakened the overload-only explanation, satisfying the falsification challenge."
        )
    if _telemetry_unavailable(evidence):
        return (
            "The telemetry source was queried successfully, but no admissible observation "
            "was available for the incident window; therefore no root-cause diagnosis was accepted."
        )
    return (
        "Decisive telemetry was unavailable and the falsification challenge remained open; "
        "no root-cause diagnosis was accepted."
    )


def _falsification_html(*, outcome: str, challenge_open: bool) -> str:
    intro = (
        "The initial overload diagnosis was rejected because workload growth and throughput loss "
        "established correlation, not causation."
    )
    if outcome == OUTCOME_RESOLVED:
        body = (
            "The challenge was satisfied only after comparison, attendance, and telemetry evidence "
            "were gathered and distinguished."
        )
    elif challenge_open:
        body = (
            "The initial causal diagnosis remained unsupported. Comparison and staffing evidence "
            "removed fallback explanations, but decisive telemetry was unavailable; therefore the "
            "challenge remained open and no diagnosis was promoted to SUPPORTED."
        )
    else:
        body = (
            "The challenge was not satisfied with admissible decisive evidence; "
            "no diagnosis was promoted to SUPPORTED."
        )
    return f"<p>{escape_report_html(intro)}</p><p>{escape_report_html(body)}</p>"


def _final_decision(*, outcome: str, evidence: PlatformProofEvidence) -> str:
    if outcome == OUTCOME_RESOLVED:
        return "Bounded H3 equipment/process degradation diagnosis accepted after falsification challenge satisfied."
    return "No root-cause diagnosis accepted."


def _evaluator_highlight_html(
    evaluation: ScenarioEvaluationResult,
    *,
    path_key: str,
) -> str:
    passed_count, total_count, overall_passed = evaluator_pass_summary(evaluation)
    verdict = "PASS" if overall_passed else "FAIL"
    verdict_css = "pass" if overall_passed else "fail"
    labels = representative_check_labels(evaluation, path_key=path_key)
    items = "".join(f"<li>{escape_report_html(label)}</li>" for label in labels)
    return (
        f'<div class="incident-evaluator-summary verdict-panel {verdict_css}">'
        f"<p><strong>{escape_report_html(EVALUATOR_DISPLAY_NAME)}</strong></p>"
        f"<p><strong>{escape_report_html(verdict)}</strong> — "
        f"{passed_count}/{total_count} checks passed</p>"
        "<p class=\"muted\">Representative checks:</p>"
        f"<ul>{items}</ul>"
        "</div>"
    )


def build_incident_report_sections(
    *,
    result: ScenarioExecutionResult,
    evaluation: ScenarioEvaluationResult,
    evidence: PlatformProofEvidence,
    variant: ScenarioVariant,
) -> tuple[RenderedReportSection, ...]:
    """Build trusted specialized sections for Scenario #1 HTML reports."""
    outcome = result.outcome
    proof_status = evidence.execution.status.value
    dispositions = _claim_dispositions(evidence)
    challenge = _challenge_resolution(evidence)
    challenge_open = challenge is ChallengeResolution.OPEN
    path_key = "resolved" if variant is ScenarioVariant.RESOLVED else "unresolved"

    outcome_cards = (
        '<div class="incident-outcome-grid">'
        '<div class="incident-outcome-card">'
        '<div class="card-label">Proof Result</div>'
        f'<div class="card-value">{render_execution_status_badge(evidence.execution.status)}</div>'
        "</div>"
        '<div class="incident-outcome-card">'
        '<div class="card-label">Incident Outcome</div>'
        f'<div class="card-value">{_outcome_badge(outcome)}</div>'
        "</div>"
        "</div>"
    )

    pass_unresolved_note = ""
    if outcome == OUTCOME_UNRESOLVED:
        pass_unresolved_note = (
            '<div class="incident-pass-unresolved-note" role="note">'
            "<p><strong>Proof PASS vs incident UNRESOLVED</strong></p>"
            "<p>PASS means the proof behaved correctly. UNRESOLVED means the available evidence "
            "did not justify accepting a root-cause diagnosis.</p>"
            "</div>"
        )

    follow_up = "<ul>" + "".join(
        f"<li>{escape_report_html(item)}</li>" for item in _FOLLOW_UP_ITEMS
    ) + "</ul>"

    investigation_html = (
        outcome_cards
        + pass_unresolved_note
        + "<h3>What happened</h3>"
        f"<p>Platform proof execution completed with proof result "
        f"<strong>{escape_report_html(proof_status)}</strong> and incident outcome "
        f"<strong>{escape_report_html(outcome)}</strong>.</p>"
        + _journey_html(resolved_path=outcome == OUTCOME_RESOLVED, challenge_open=challenge_open)
        + "<h3>Initial hypothesis</h3>"
        f"<p>{escape_report_html(_INITIAL_HYPOTHESIS)}</p>"
        + "<h3>Independent challenge</h3>"
        f"<p>{escape_report_html(_CHALLENGE_SUMMARY)}</p>"
        + "<h3>Follow-up evidence</h3>"
        + follow_up
        + "<h3>Final hypothesis state</h3>"
        + _hypothesis_cards(dispositions)
        + "<h3>Why this path is "
        + escape_report_html(outcome)
        + "</h3>"
        f"<p>{escape_report_html(_why_resolved_or_unresolved(outcome=outcome, evidence=evidence, challenge_open=challenge_open))}</p>"
        + "<h3>Final decision</h3>"
        f"<p>{escape_report_html(_final_decision(outcome=outcome, evidence=evidence))}</p>"
        + "<h3>Evaluator result</h3>"
        + _evaluator_highlight_html(evaluation, path_key=path_key)
    )

    defensibility_html = _falsification_html(outcome=outcome, challenge_open=challenge_open)

    return (
        RenderedReportSection(
            section_id="incident-investigation-result",
            title="Investigation Result",
            html=investigation_html,
        ),
        RenderedReportSection(
            section_id="incident-defensibility",
            title="Why this result is defensible",
            html=defensibility_html,
        ),
    )
