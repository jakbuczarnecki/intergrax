# Â© Artur Czarnecki. All rights reserved.

"""Representative PlatformProofEvidence fixtures for HTML renderer tests (PP-REPORT-3)."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEdgeEvidence,
    ArchitectureEvidence,
    ConclusionEvidence,
    DatasetEnvironmentEvidence,
    EnvironmentEvidence,
    EvaluatorCheckEvidence,
    EvaluatorSummaryEvidence,
    EvidenceEdge,
    EvidenceGraphEvidence,
    EvidenceNode,
    EvidenceNodeKind,
    EvidenceRelationship,
    ExecutionMetadataEvidence,
    FailureClassification,
    FailureEvidence,
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
    ReportSafeField,
    ReportSafePayload,
    ReportSafeVisibility,
    ReproductionEvidence,
    ScenarioEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    redacted_report_safe_text,
    sanitized_runtime_report_safe_text,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile

_TS = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)
_TS_END = datetime(2026, 8, 21, 12, 5, 0, tzinfo=UTC)
_REVISION = "abc123def456"
_EXECUTION_ID = "exec-fixture-001"


def _participant_llm() -> ParticipantEvidence:
    return ParticipantEvidence(
        participant_id="llm",
        name="LLM Provider",
        implementation="OpenAI",
        version_or_model="gpt-test",
        role="Planning and answers",
        participant_class=ParticipantClass.EXTERNAL_VENDOR,
    )


def _participant_runtime() -> ParticipantEvidence:
    return ParticipantEvidence(
        participant_id="runtime",
        name="Intergrax Nexus Runtime",
        implementation="Intergrax",
        version_or_model="3.12.0",
        role="Bounded tool loop",
        participant_class=ParticipantClass.PLATFORM,
    )


def _participant_db() -> ParticipantEvidence:
    return ParticipantEvidence(
        participant_id="postgres",
        name="PostgreSQL",
        implementation="PostgreSQL",
        version_or_model="16",
        role="Fixture dataset store",
        participant_class=ParticipantClass.CONTROLLED_FIXTURE,
    )


def _base_claim() -> ProofClaimEvidence:
    return ProofClaimEvidence(
        claim="Platform can drive evidence-dependent tool calls using real observations.",
        user_relevance="Validates bounded iterative tool runtime for investigative workflows.",
        success_criteria=("Minimum tool calls succeed", "Evidence chain preserved"),
        falsification_criteria=("Unsupported causal claim accepted",),
        excluded_claims=(
            "Production readiness",
            "Universal provider compatibility",
            "Commercial validation",
        ),
    )


def _base_architecture() -> ArchitectureEvidence:
    return ArchitectureEvidence(
        participants=(_participant_llm(), _participant_runtime(), _participant_db()),
        edges=(
            ArchitectureEdgeEvidence(
                from_participant="llm",
                to_participant="runtime",
                relationship="tool planning",
            ),
            ArchitectureEdgeEvidence(
                from_participant="runtime",
                to_participant="postgres",
                relationship="SQL query",
            ),
        ),
    )


def _base_environment() -> EnvironmentEvidence:
    return EnvironmentEvidence(
        dataset=DatasetEnvironmentEvidence(
            dataset_id="fixture-parcel-events",
            dataset_version="v1",
            row_count=1200,
            seed=42,
            fingerprint_sha256="a" * 64,
            infrastructure_identity="docker-postgres-16",
            access_mode="read-only",
            ground_truth_checks=("North hub anomaly present",),
            information_exposed_to_model=("SQL tool results only",),
        )
    )


def _trace_steps() -> tuple[ProofExecutionStep, ...]:
    return (
        ProofExecutionStep(
            step_index=0,
            step_id="step-dataset",
            purpose=proof_authored_report_safe_text("Prepare deterministic dataset"),
            evidence_basis_ids=(),
            action=proof_authored_report_safe_text("Materialize and verify fingerprint"),
            input=ReportSafePayload(
                summary=proof_authored_report_safe_text("Dataset bootstrap"),
                fields=(
                    ReportSafeField(
                        name="row_count",
                        visibility=ReportSafeVisibility.REPORT_SAFE,
                        value=1200,
                    ),
                ),
            ),
            observation=ReportSafePayload(
                summary=proof_authored_report_safe_text("Dataset verified"),
            ),
            evidence_created_ids=("evidence-dataset",),
            status=ProofStepExecutionStatus.OK,
        ),
        ProofExecutionStep(
            step_index=1,
            step_id="step-query",
            purpose=proof_authored_report_safe_text("Inspect regional segment"),
            evidence_basis_ids=("evidence-dataset",),
            action=proof_authored_report_safe_text("Execute bounded SQL query"),
            input=ReportSafePayload(
                summary=proof_authored_report_safe_text("SQL query arguments"),
                fields=(
                    ReportSafeField(
                        name="sql",
                        visibility=ReportSafeVisibility.REPORT_SAFE,
                        value=proof_authored_report_safe_text(
                            "SELECT region, AVG(delayed::int) FROM proof.parcel_events GROUP BY region"
                        ),
                    ),
                ),
            ),
            observation=ReportSafePayload(
                summary=explicit_runtime_report_safe_text("North region delay rate elevated"),
            ),
            evidence_created_ids=("evidence-query-1",),
            status=ProofStepExecutionStatus.OK,
        ),
    )


def _evidence_graph() -> EvidenceGraphEvidence:
    return EvidenceGraphEvidence(
        nodes=(
            EvidenceNode(
                evidence_id="evidence-dataset",
                kind=EvidenceNodeKind.DATASET,
                label="Dataset",
                summary=proof_authored_report_safe_text("Verified fixture dataset"),
                producing_step_id="step-dataset",
            ),
            EvidenceNode(
                evidence_id="evidence-query-1",
                kind=EvidenceNodeKind.TOOL_RESULT,
                label="Regional query",
                summary=explicit_runtime_report_safe_text("North region delay rate elevated"),
                producing_step_id="step-query",
            ),
        ),
        edges=(
            EvidenceEdge(
                from_evidence_id="evidence-dataset",
                to_evidence_id="evidence-query-1",
                relationship=EvidenceRelationship.EVIDENCE_BASIS,
            ),
        ),
    )


def _base_provenance(*, proof_id: str = "GENERIC-PLATFORM-PROOF") -> ProvenanceEvidence:
    return ProvenanceEvidence(
        proof_id=proof_id,
        source_revision=_REVISION,
        generated_at=_TS_END,
        execution_id=_EXECUTION_ID,
        evidence_checksum="f" * 64,
        artifact_identity=f"intergrax.platform_proof_evidence.v3:{proof_id}:{_EXECUTION_ID}:20260821T120500Z",
    )


def _base_reproduction() -> ReproductionEvidence:
    return ReproductionEvidence(
        source_revision=_REVISION,
        command="uv run python -m scripts.proof.intergrax_proof_runner --profile full",
        prerequisites=("Docker available", "Provider credentials configured"),
        required_env_variable_names=("INTERGRAX_LLM_PROVIDER",),
        dataset_fingerprint_sha256="a" * 64,
    )


def _base_identity(*, proof_id: str = "GENERIC-PLATFORM-PROOF") -> ProofIdentityEvidence:
    return ProofIdentityEvidence(
        proof_id=proof_id,
        title="Generic Platform Proof",
        domains_exercised=("platform",),
        proof_version="1.0.0",
        source_revision=_REVISION,
        execution_profile=ProofProfile.FULL,
    )


def _base_execution(*, status: ProofEvidenceExecutionStatus) -> ExecutionMetadataEvidence:
    return ExecutionMetadataEvidence(
        status=status,
        started_at=_TS,
        finished_at=_TS_END,
        duration_ms=300_000,
        platform="linux",
        runtime_version="3.12.0",
        source_dirty=False,
    )


def build_pass_evidence() -> PlatformProofEvidence:
    steps = _trace_steps()
    return PlatformProofEvidence(
        proof_identity=_base_identity(),
        execution=_base_execution(status=ProofEvidenceExecutionStatus.PASS),
        claim=_base_claim(),
        architecture=_base_architecture(),
        participants=_base_architecture().participants,
        environment=_base_environment(),
        scenarios=(
            ScenarioEvidence(
                scenario_id="scenario-a",
                title="Regional anomaly",
                question="Which region shows elevated delays?",
                expected_behavior="Identify North segment anomaly",
                falsification_condition="Volume-only explanation accepted",
                execution_status=ProofEvidenceExecutionStatus.PASS,
                metrics=(MetricEvidence(name="successful_tool_calls", value=2),),
                steps=steps,
                final_output=FinalOutputEvidence(
                    present=True,
                    content=explicit_runtime_report_safe_text(
                        "North delays driven by express long_haul segment."
                    ),
                ),
                evaluator=EvaluatorSummaryEvidence(
                    passed=True,
                    checks=(
                        EvaluatorCheckEvidence(
                            check_id="min-tool-calls",
                            label="Minimum successful tool calls",
                            passed=True,
                            explanation=proof_authored_report_safe_text("2 tool calls succeeded"),
                            evidence_ids=("evidence-query-1",),
                        ),
                    ),
                ),
            ),
        ),
        evidence_graph=_evidence_graph(),
        final_output=FinalOutputEvidence(
            present=True,
            content=explicit_runtime_report_safe_text(
                "North delays driven by express long_haul segment."
            ),
        ),
        evaluator=EvaluatorSummaryEvidence(
            passed=True,
            checks=(
                EvaluatorCheckEvidence(
                    check_id="overall",
                    label="Scenario A passed",
                    passed=True,
                    explanation=proof_authored_report_safe_text("All checks passed"),
                ),
            ),
        ),
        limitations=("Single provider run", "Bounded row cap"),
        conclusion=ConclusionEvidence(
            supported_conclusions=("Evidence-dependent tool chain demonstrated",),
            unsupported_conclusions=("Production readiness",),
            open_questions=("Multi-provider validation",),
        ),
        reproduction=_base_reproduction(),
        provenance=_base_provenance(),
    )


def build_fail_evidence() -> PlatformProofEvidence:
    return PlatformProofEvidence(
        proof_identity=_base_identity(proof_id="GENERIC-PROOF-FAIL"),
        execution=_base_execution(status=ProofEvidenceExecutionStatus.FAIL),
        claim=_base_claim(),
        architecture=_base_architecture(),
        participants=_base_architecture().participants,
        environment=_base_environment(),
        scenarios=(
            ScenarioEvidence(
                scenario_id="scenario-b",
                title="Causation check",
                question="Is direct causation supported?",
                expected_behavior="Reject unsupported causation",
                falsification_condition="Direct causation asserted",
                execution_status=ProofEvidenceExecutionStatus.FAIL,
                steps=_trace_steps(),
                final_output=FinalOutputEvidence(
                    present=True,
                    content=explicit_runtime_report_safe_text(
                        "Weight directly causes all network delays."
                    ),
                ),
                evaluator=EvaluatorSummaryEvidence(
                    passed=False,
                    checks=(
                        EvaluatorCheckEvidence(
                            check_id="no-causation",
                            label="Reject direct causation",
                            passed=False,
                            explanation=proof_authored_report_safe_text(
                                "Model asserted unsupported causation"
                            ),
                        ),
                    ),
                    failure_reasons=("Unsupported causal claim",),
                ),
                failure=FailureEvidence(
                    classification=FailureClassification.MODEL_BEHAVIOR_FAILURE,
                    boundary="evaluator",
                    message=proof_authored_report_safe_text("Model asserted unsupported causation"),
                    completed_milestones=("dataset verified", "tool calls completed"),
                    failed_milestone="evaluator causation check",
                ),
            ),
        ),
        evidence_graph=_evidence_graph(),
        final_output=FinalOutputEvidence(
            present=True,
            content=explicit_runtime_report_safe_text(
                "Weight directly causes all network delays."
            ),
        ),
        evaluator=EvaluatorSummaryEvidence(
            passed=False,
            failure_reasons=("Scenario B failed",),
        ),
        limitations=("Single model",),
        conclusion=ConclusionEvidence(
            unsupported_conclusions=("Causal claim validated",),
        ),
        reproduction=_base_reproduction(),
        provenance=_base_provenance(proof_id="GENERIC-PROOF-FAIL"),
        failure=FailureEvidence(
            classification=FailureClassification.MODEL_BEHAVIOR_FAILURE,
            boundary="evaluator",
            message=proof_authored_report_safe_text("Claim not demonstrated"),
            completed_milestones=("dataset verified",),
            failed_milestone="evaluator",
        ),
    )


def build_blocked_evidence() -> PlatformProofEvidence:
    return PlatformProofEvidence(
        proof_identity=_base_identity(proof_id="GENERIC-PROOF-BLOCKED"),
        execution=_base_execution(status=ProofEvidenceExecutionStatus.BLOCKED),
        claim=_base_claim(),
        architecture=_base_architecture(),
        participants=_base_architecture().participants,
        environment=EnvironmentEvidence(),
        scenarios=(),
        limitations=("Execution blocked before scenarios",),
        conclusion=ConclusionEvidence(),
        reproduction=_base_reproduction(),
        provenance=_base_provenance(proof_id="GENERIC-PROOF-BLOCKED"),
        failure=FailureEvidence(
            classification=FailureClassification.BLOCKED_CONFIGURATION,
            boundary="configuration gate",
            message=proof_authored_report_safe_text("Missing INTERGRAX_LLM_PROVIDER"),
            completed_milestones=("manifest loaded",),
            skipped_not_reached=("adapter construction", "tool loop", "evaluator"),
        ),
    )


def build_crash_evidence() -> PlatformProofEvidence:
    return PlatformProofEvidence(
        proof_identity=_base_identity(proof_id="GENERIC-PROOF-CRASH"),
        execution=_base_execution(status=ProofEvidenceExecutionStatus.CRASH),
        claim=_base_claim(),
        architecture=_base_architecture(),
        participants=_base_architecture().participants,
        environment=_base_environment(),
        scenarios=(),
        limitations=("Abnormal termination",),
        conclusion=ConclusionEvidence(),
        reproduction=_base_reproduction(),
        provenance=_base_provenance(proof_id="GENERIC-PROOF-CRASH"),
        failure=FailureEvidence(
            classification=FailureClassification.UNKNOWN,
            boundary="provider request",
            message=sanitized_runtime_report_safe_text("Provider request failed unexpectedly"),
            completed_milestones=("dataset verified", "adapter constructed"),
            failed_milestone="provider request",
            skipped_not_reached=("model response", "evaluator"),
        ),
    )


def build_injection_evidence() -> PlatformProofEvidence:
    evidence = build_pass_evidence()
    return evidence.model_copy(
        update={
            "claim": evidence.claim.model_copy(
                update={
                    "claim": '<script>alert(1)</script> injected claim text',
                }
            ),
        }
    )


def build_redacted_payload_evidence() -> PlatformProofEvidence:
    steps = (
        ProofExecutionStep(
            step_index=0,
            step_id="step-redacted",
            purpose=proof_authored_report_safe_text("Access credential field"),
            evidence_basis_ids=(),
            action=proof_authored_report_safe_text("Read configuration"),
            input=ReportSafePayload(
                summary=redacted_report_safe_text(),
                fields=(
                    ReportSafeField(
                        name="api_token",
                        visibility=ReportSafeVisibility.REDACTED,
                        value=redacted_report_safe_text(),
                    ),
                ),
            ),
            observation=ReportSafePayload(
                summary=proof_authored_report_safe_text("Configuration read attempted"),
            ),
            status=ProofStepExecutionStatus.OK,
        ),
    )
    scenario = build_pass_evidence().scenarios[0].model_copy(update={"steps": steps})
    graph = EvidenceGraphEvidence()
    return build_pass_evidence().model_copy(
        update={
            "scenarios": (scenario,),
            "evidence_graph": graph,
        }
    )


def build_multi_scenario_evidence() -> PlatformProofEvidence:
    base = build_pass_evidence()
    second = base.scenarios[0].model_copy(
        update={
            "scenario_id": "scenario-b",
            "title": "Secondary scenario",
            "execution_status": ProofEvidenceExecutionStatus.PASS,
        }
    )
    return base.model_copy(update={"scenarios": (base.scenarios[0], second)})
