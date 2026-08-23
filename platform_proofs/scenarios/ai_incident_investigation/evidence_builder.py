# © Artur Czarnecki. All rights reserved.

"""Project runtime facts into PlatformProofEvidence v3."""

from __future__ import annotations

from datetime import UTC, datetime

from intergrax.contracts.evidence_claims import EvidenceClaimSet
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
    ReportSafeTextSourceKind,
    ReproductionEvidence,
    ScenarioEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    project_evidence_claim_set,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile
from platform_proofs.scenarios.ai_incident_investigation.scenario import ScenarioExecutionResult

PROOF_ID = "SCENARIO-AI-INCIDENT-INVESTIGATION-SKELETON"
PROOF_VERSION = "full-1-resolved-0.2.0"


def build_platform_proof_evidence(
    result: ScenarioExecutionResult,
    *,
    source_revision: str,
    finished_at: datetime | None = None,
) -> PlatformProofEvidence:
    finished = finished_at or datetime.now(tz=UTC)
    started = finished
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
            summary=proof_authored_report_safe_text("observable tool result"),
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
        step_index=0,
        step_id="step-investigate",
        purpose=proof_authored_report_safe_text("Investigate incident via platform graph path"),
        evidence_basis_ids=(),
        action=proof_authored_report_safe_text("Investigator agent with ToolRuntime follow-up"),
        observation=ReportSafePayload(
            summary=explicit_runtime_report_safe_text(result.terminal_summary),
        ),
        evidence_created_ids=tuple(str(node["evidence_id"]) for node in result.evidence_nodes),
        status=ProofStepExecutionStatus.OK,
    )

    execution_id = f"incident-skeleton-{int(started.timestamp())}"
    scenario = ScenarioEvidence(
        scenario_id="resolved_full_evidence_world",
        title="RESOLVED FULL — H1/H2/H3 evidence world",
        question=(
            "Can workload-throughput correlation become accepted diagnosis without "
            "comparison, staffing resolution, and telemetry follow-up?"
        ),
        expected_behavior=(
            "Reject H1 overload; resolve H2 staffing conflict; gather comparison and "
            "telemetry; accept bounded H3 diagnosis"
        ),
        falsification_condition=(
            "Unsupported overload diagnosis accepted without distinguishing evidence"
        ),
        execution_status=ProofEvidenceExecutionStatus.PASS,
        metrics=(
            MetricEvidence(name="tool_invocations", value=result.tool_invocations),
            MetricEvidence(name="evaluator_loop_iterations", value=result.evaluator_loop_iterations),
        ),
        steps=(step,),
        final_output=FinalOutputEvidence(
            present=True,
            content=explicit_runtime_report_safe_text(result.terminal_summary),
        ),
    )

    return PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=PROOF_ID,
            title="AI Incident Investigation — full RESOLVED evidence world",
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
            user_relevance="FULL-1 proves RESOLVED path with H1/H2/H3 adversarial evidence.",
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
            "FULL-1 RESOLVED evidence world only — UNRESOLVED path pending FULL-2.",
            "Not accepted for public proof publication.",
        ),
        conclusion=ConclusionEvidence(
            supported_conclusions=(
                "RESOLVED path with H1/H2/H3 adversarial evidence exercised",
            ),
            unsupported_conclusions=(
                "UNRESOLVED insufficient-evidence path",
                "Public proof publication",
            ),
            open_questions=("FULL-2 UNRESOLVED terminal scenario",),
        ),
        reproduction=ReproductionEvidence(
            source_revision=source_revision,
            command="uv run pytest tests/unit/platform_proofs/scenarios/ai_incident_investigation/",
            prerequisites=("Python 3.12", "uv"),
        ),
        provenance=ProvenanceEvidence(
            proof_id=PROOF_ID,
            source_revision=source_revision,
            generated_at=finished,
            execution_id=execution_id,
            artifact_identity=f"intergrax.platform_proof_evidence.v3:{PROOF_ID}:{execution_id}",
        ),
    )
