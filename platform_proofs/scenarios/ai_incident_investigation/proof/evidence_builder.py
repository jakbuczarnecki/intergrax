# © Artur Czarnecki. All rights reserved.

"""Project runtime facts into PlatformProofEvidence v3."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.evidence_claims import EvidenceClaimSet
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator import ScenarioEvaluationResult
from platform_proofs.scenarios.ai_incident_investigation.proof.reproduction import (
    CANONICAL_REPRODUCTION_PREREQUISITES,
    PROOF_ID,
    canonical_reproduction_shell_command,
)
from platform_proofs.scenarios.ai_incident_investigation.proof.evaluator_evidence import (
    project_scenario_evaluation_to_evidence,
)
from platform_proofs.scenarios.ai_incident_investigation.fixtures.incidents import ScenarioVariant
from platform_proofs.scenarios.ai_incident_investigation.application.investigator_agent import (
    TELEMETRY_EVIDENCE_ID,
)
from platform_proofs.scenarios.ai_incident_investigation.application.scenario import (
    OUTCOME_RESOLVED,
    ScenarioExecutionResult,
)
from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEvidence,
    ConclusionEvidence,
    EnvironmentEvidence,
    EvidenceGraphEvidence,
    EvidenceNode,
    EvidenceNodeKind,
    ExecutionMetadataEvidence,
    FinalOutputEvidence,
    MetricEvidence,
    ParticipantClass,
    ParticipantEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofExecutionStep,
    ProofIdentityEvidence,
    ProofStepExecutionStatus,
    ProvenanceEvidence,
    ReportSafePayload,
    ReportSafeText,
    ReportSafeTextSourceKind,
    ReproductionEvidence,
    ScenarioEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    project_evidence_claim_set,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile

PROOF_VERSION = "full-1-resolved-full-2-unresolved-0.3.0"

EVIDENCE_RESOLVED_FILENAME = "evidence-resolved.json"
EVIDENCE_UNRESOLVED_FILENAME = "evidence-unresolved.json"
REPORT_RESOLVED_FILENAME = "report-resolved.html"
REPORT_UNRESOLVED_FILENAME = "report-unresolved.html"

_SCENARIO_METADATA: dict[ScenarioVariant, dict[str, str]] = {
    ScenarioVariant.RESOLVED: {
        "scenario_id": "resolved_full_evidence_world",
        "scenario_title": "RESOLVED FULL — H1/H2/H3 evidence world",
        "proof_title": "AI Incident Investigation — RESOLVED evidence path",
        "expected_behavior": (
            "Reject H1 overload; resolve H2 staffing conflict; gather comparison and "
            "telemetry; accept bounded H3 diagnosis"
        ),
        "supported_conclusion": (
            "RESOLVED path with decisive telemetry and bounded H3 diagnosis"
        ),
        "execution_suffix": "resolved",
    },
    ScenarioVariant.UNRESOLVED: {
        "scenario_id": "unresolved_full_evidence_world",
        "scenario_title": "UNRESOLVED FULL — telemetry unavailable evidence world",
        "proof_title": "AI Incident Investigation — UNRESOLVED evidence path",
        "expected_behavior": (
            "Reject H1 overload; resolve H2 staffing conflict; gather comparison; "
            "accept epistemic UNRESOLVED when telemetry is unavailable for incident window"
        ),
        "supported_conclusion": (
            "UNRESOLVED path when decisive telemetry is unavailable for incident window"
        ),
        "execution_suffix": "unresolved",
    },
}


def _telemetry_evidence_summary(payload: object) -> ReportSafeText:
    if not isinstance(payload, dict):
        return proof_authored_report_safe_text("observable tool result")
    availability = payload.get("availability")
    if availability == "unavailable":
        reason = str(payload.get("unavailability_reason", "no_observation_for_window"))
        return proof_authored_report_safe_text(
            "Telemetry source queried successfully; observation unavailable for incident window "
            f"({reason})"
        )
    if availability == "available":
        signal_state = str(payload.get("signal_state", "unknown"))
        return proof_authored_report_safe_text(
            f"Telemetry available; station signal degraded ({signal_state})"
        )
    return proof_authored_report_safe_text("observable tool result")


def _evidence_node_summary(node: dict[str, object]) -> ReportSafeText:
    evidence_id = str(node.get("evidence_id", ""))
    if evidence_id == str(TELEMETRY_EVIDENCE_ID) or "telemetry" in evidence_id:
        return _telemetry_evidence_summary(node.get("payload"))
    return proof_authored_report_safe_text("observable tool result")


def _planner_execution_steps(
    result: ScenarioExecutionResult,
) -> tuple[ProofExecutionStep, ...]:
    steps: list[ProofExecutionStep] = []
    for index, decision in enumerate(result.planner_decisions):
        selected = decision.get("selected_tool_ids") or []
        tool_id = str(selected[0]) if selected else "unknown_tool"
        basis_ids = tuple(str(item) for item in decision.get("evidence_basis_evidence_ids") or ())
        created_ids = tuple(
            str(node["evidence_id"])
            for node in result.evidence_nodes
            if str(node.get("source_tool_id", "")) == tool_id
        )
        steps.append(
            ProofExecutionStep(
                step_index=index,
                step_id=f"step-planner-{index + 1}",
                purpose=explicit_runtime_report_safe_text(
                    str(decision.get("objective", "gather incident evidence"))
                ),
                evidence_basis_ids=basis_ids,
                action=explicit_runtime_report_safe_text(f"Planner selected {tool_id}"),
                evidence_created_ids=created_ids,
                status=ProofStepExecutionStatus.OK,
            )
        )
    if not steps:
        return ()
    return tuple(steps)


def build_platform_proof_evidence(
    result: ScenarioExecutionResult,
    *,
    evaluation: ScenarioEvaluationResult,
    variant: ScenarioVariant,
    source_revision: str,
    finished_at: datetime | None = None,
) -> PlatformProofEvidence:
    finished = finished_at or datetime.now(tz=UTC)
    started = finished
    meta = _SCENARIO_METADATA[variant]
    claim_set = EvidenceClaimSet.model_validate(result.claim_set)
    projected_claims = project_evidence_claim_set(
        claim_set,
        text_source=ReportSafeTextSourceKind.RUNTIME_EXPLICIT,
    )

    evidence_nodes = tuple(
        EvidenceNode(
            evidence_id=str(node["evidence_id"]),
            kind=EvidenceNodeKind.TOOL_RESULT,
            label=str(node.get("label", "tool observation")),
            summary=_evidence_node_summary(node),
            producing_step_id="step-investigate",
        )
        for node in result.evidence_nodes
    )

    participant_platform = ParticipantEvidence(
        participant_id="nexus_runtime",
        name="Intergrax Nexus Runtime",
        implementation="Intergrax",
        version_or_model="development",
        role="Graph execution, ToolRuntime, Critic",
        participant_class=ParticipantClass.PLATFORM,
    )
    participant_fixture = ParticipantEvidence(
        participant_id="synthetic_fixture",
        name="Synthetic manufacturing fixture",
        implementation="scenario-local",
        version_or_model=PROOF_VERSION,
        role="Controlled operational observations",
        participant_class=ParticipantClass.CONTROLLED_FIXTURE,
    )

    step = ProofExecutionStep(
        step_index=len(result.planner_decisions),
        step_id="step-investigate",
        purpose=explicit_runtime_report_safe_text(
            result.evidence_gathering_stop_reason
            or "Investigate incident via autonomous bounded tool loop"
        ),
        evidence_basis_ids=(),
        action=explicit_runtime_report_safe_text(
            "Investigator agent autonomous evidence gathering and deterministic diagnosis"
        ),
        observation=ReportSafePayload(
            summary=explicit_runtime_report_safe_text(result.terminal_summary),
        ),
        evidence_created_ids=tuple(str(node["evidence_id"]) for node in result.evidence_nodes),
        status=ProofStepExecutionStatus.OK,
    )
    execution_steps = _planner_execution_steps(result) + (step,)

    execution_id = f"incident-{meta['execution_suffix']}-{int(started.timestamp())}"
    paired_note = (
        "Paired UNRESOLVED canonical evidence exists in evidence-unresolved.json"
        if variant is ScenarioVariant.RESOLVED
        else "Paired RESOLVED canonical evidence exists in evidence-resolved.json"
    )

    evaluator_evidence = project_scenario_evaluation_to_evidence(evaluation)

    scenario = ScenarioEvidence(
        scenario_id=meta["scenario_id"],
        title=meta["scenario_title"],
        question=(
            "Can workload-throughput correlation become accepted diagnosis without "
            "comparison, staffing resolution, and telemetry follow-up?"
        ),
        expected_behavior=meta["expected_behavior"],
        falsification_condition=(
            "Unsupported overload diagnosis accepted without distinguishing evidence"
        ),
        execution_status=ProofEvidenceExecutionStatus.PASS,
        metrics=(
            MetricEvidence(name="tool_invocations", value=result.tool_invocations),
            MetricEvidence(name="evaluator_loop_iterations", value=result.evaluator_loop_iterations),
            MetricEvidence(
                name="scenario_outcome",
                value=1 if result.outcome == OUTCOME_RESOLVED else 0,
            ),
        ),
        steps=execution_steps,
        final_output=FinalOutputEvidence(
            present=True,
            content=explicit_runtime_report_safe_text(result.terminal_summary),
        ),
        evaluator=evaluator_evidence,
    )

    unsupported_conclusions: tuple[str, ...] = ("Public proof publication",)
    if variant is ScenarioVariant.RESOLVED:
        unsupported_conclusions = (
            "Public proof publication",
            "UNRESOLVED path proven solely by this artifact's claim graph",
        )
    else:
        unsupported_conclusions = (
            "Public proof publication",
            "RESOLVED path proven solely by this artifact's claim graph",
        )

    return PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=PROOF_ID,
            title=meta["proof_title"],
            domains_exercised=("EXECUTION", "TOOLS", "CRITIC", "EVIDENCE"),
            proof_version=PROOF_VERSION,
            source_revision=source_revision,
            execution_profile=ProofProfile.QUICK,
        ),
        execution=ExecutionMetadataEvidence(
            status=ProofEvidenceExecutionStatus.PASS,
            started_at=started,
            finished_at=finished,
            platform="intergrax",
        ),
        claim=ProofClaimEvidence(
            claim="No material incident diagnosis without auditable evidence and falsification.",
            user_relevance=(
                "FULL-1/FULL-2 prove RESOLVED and UNRESOLVED paths with H1/H2/H3 adversarial evidence."
            ),
            success_criteria=(
                "ToolRuntime exercised",
                "Critic challenge and revision bounded",
                "Comparison and staffing conflict handled",
            ),
            falsification_criteria=("Unsupported causal diagnosis accepted",),
            excluded_claims=("Full manufacturing dataset", "Public proof established"),
        ),
        architecture=ArchitectureEvidence(
            participants=(participant_platform, participant_fixture),
            edges=(),
        ),
        participants=(participant_platform, participant_fixture),
        environment=EnvironmentEvidence(),
        scenarios=(scenario,),
        evidence_graph=EvidenceGraphEvidence(nodes=evidence_nodes, edges=()),
        evidence_claims=projected_claims,
        final_output=FinalOutputEvidence(
            present=True,
            content=explicit_runtime_report_safe_text(result.terminal_summary),
        ),
        limitations=(
            (
                "This artifact demonstrates the RESOLVED path only; paired UNRESOLVED "
                "canonical evidence is produced separately."
                if variant is ScenarioVariant.RESOLVED
                else "This artifact demonstrates the UNRESOLVED path only; paired RESOLVED "
                "canonical evidence is produced separately."
            ),
            paired_note,
            "Not accepted for public proof publication.",
        ),
        evaluator=evaluator_evidence,
        conclusion=ConclusionEvidence(
            supported_conclusions=(meta["supported_conclusion"],),
            unsupported_conclusions=unsupported_conclusions,
            open_questions=("Accepted public proof run",),
        ),
        reproduction=ReproductionEvidence(
            source_revision=source_revision,
            command=canonical_reproduction_shell_command(),
            prerequisites=CANONICAL_REPRODUCTION_PREREQUISITES,
        ),
        provenance=ProvenanceEvidence(
            proof_id=PROOF_ID,
            source_revision=source_revision,
            generated_at=finished,
            execution_id=execution_id,
            artifact_identity=f"intergrax.platform_proof_evidence.v3:{PROOF_ID}:{execution_id}",
        ),
    )
