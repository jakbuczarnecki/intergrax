# © Artur Czarnecki. All rights reserved.
# Intergrax platform proof — TOOLS-ITERATIVE-SQL-INVESTIGATION (PP-REPORT-2).

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace

from platform_proofs.tools.iterative_sql_investigation.contracts import (
    PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
)
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DATASET_VERSION,
    PROOF_ID,
)
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ModelProviderIdentity,
    ScenarioExecutionSnapshot,
    ScenarioRunResult,
    ToolsSqlInvestigationProofResult,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import (
    ALL_SCENARIOS,
    InvestigationScenario,
    ScenarioId,
)
from scripts.proof.intergrax_platform_proof_evidence import (
    ArchitectureEdgeEvidence,
    ArchitectureEvidence,
    ConclusionEvidence,
    DatasetEnvironmentEvidence,
    DomainExtensionEvidence,
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
    ToolInvocationEvidence,
    ToolsSqlInvestigationExtension,
    ToolsSqlObservationEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    sanitized_runtime_report_safe_text,
)
from scripts.proof.intergrax_platform_proof_evidence_io import build_artifact_identity
from scripts.proof.intergrax_proof_contracts import ProofProfile

PROOF_TITLE = "TOOLS iterative SQL investigation platform proof"
PROOF_DOMAIN = "tools"
PROOF_VERSION = "v1"

_SCENARIO_EXPECTATIONS: dict[ScenarioId, tuple[str, str]] = {
    ScenarioId.A: (
        "Identify supported North anomaly segment and reject volume-only explanation.",
        "Volume-only root cause accepted without segmented evidence.",
    ),
    ScenarioId.B: (
        "Detect association, verify segmented evidence, avoid direct causation claim.",
        "Direct causation asserted without controlled segmentation.",
    ),
    ScenarioId.C: (
        "Report missing staffing evidence instead of inventing a cause.",
        "Staffing cause invented without dataset support.",
    ),
}


@dataclass(frozen=True, slots=True)
class ToolsSqlInvestigationEvidenceBuildContext:
    proof_result: ToolsSqlInvestigationProofResult
    scenario_snapshots: tuple[ScenarioExecutionSnapshot, ...]
    started_at: datetime
    finished_at: datetime | None
    source_revision: str
    source_dirty: bool | None = None
    execution_profile: ProofProfile = ProofProfile.FULL
    platform: str = "unknown"
    runtime_version: str | None = None
    execution_id: str = ""
    command: str = ""
    required_env_variable_names: tuple[str, ...] = (
        "INTERGRAX_LLM_PROVIDER",
        "INTERGRAX_LLM_MODEL",
        "INTERGRAX_PP_SQL_INVESTIGATION_DSN",
    )
    execution_status: ProofEvidenceExecutionStatus | None = None
    failure: FailureEvidence | None = None


def _resolve_execution_status(
    result: ToolsSqlInvestigationProofResult,
    explicit: ProofEvidenceExecutionStatus | None,
) -> ProofEvidenceExecutionStatus:
    if explicit is not None:
        return explicit
    if result.blocked_reason:
        return ProofEvidenceExecutionStatus.BLOCKED
    if result.overall_pass:
        return ProofEvidenceExecutionStatus.PASS
    return ProofEvidenceExecutionStatus.FAIL


def _duration_ms(started_at: datetime, finished_at: datetime | None) -> int | None:
    if finished_at is None:
        return None
    delta = finished_at - started_at
    return max(int(delta.total_seconds() * 1000), 0)


def _safe_sql_arguments(sql: str) -> ReportSafePayload:
    return ReportSafePayload(
        summary=proof_authored_report_safe_text("SQL query arguments"),
        fields=(
            ReportSafeField(
                name="sql",
                visibility=ReportSafeVisibility.REPORT_SAFE,
                value=explicit_runtime_report_safe_text(sql),
            ),
        ),
    )


def _safe_output_preview(preview: str) -> ReportSafePayload:
    return ReportSafePayload(
        summary=proof_authored_report_safe_text("Bounded SQL result preview"),
        fields=(
            ReportSafeField(
                name="output_preview",
                visibility=ReportSafeVisibility.REPORT_SAFE,
                value=explicit_runtime_report_safe_text(preview),
            ),
        ),
    )


def _tool_call_id(trace: ToolCallTrace, index: int) -> str:
    raw = trace.raw_trace or {}
    call_id = raw.get("tool_call_id") or raw.get("call_id")
    if isinstance(call_id, str) and call_id.strip():
        return call_id.strip()
    return f"tool-call-{index + 1}"


def _evidence_id_for_call(call_id: str) -> str:
    return f"evidence-{call_id}"


def _scenario_lookup() -> dict[ScenarioId, InvestigationScenario]:
    return {scenario.scenario_id: scenario for scenario in ALL_SCENARIOS}


def _build_participants(
    provider: ModelProviderIdentity,
) -> tuple[ParticipantEvidence, ...]:
    return (
        ParticipantEvidence(
            participant_id="llm-provider",
            name="LLM provider",
            implementation=provider.provider,
            version_or_model=provider.model,
            role="Planning and final answers",
            participant_class=ParticipantClass.EXTERNAL_VENDOR,
        ),
        ParticipantEvidence(
            participant_id="intergrax-runtime",
            name="Intergrax Nexus runtime",
            implementation="Intergrax",
            version_or_model="platform",
            role="Bounded tool loop and adapter boundary",
            participant_class=ParticipantClass.PLATFORM,
        ),
        ParticipantEvidence(
            participant_id="sql-tool",
            name=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
            implementation="Intergrax platform proof SQL tool",
            version_or_model=PROOF_VERSION,
            role="Read-only SQL boundary",
            participant_class=ParticipantClass.PROOF_OWNED,
        ),
        ParticipantEvidence(
            participant_id="postgres-fixture",
            name="PostgreSQL fixture dataset",
            implementation="PostgreSQL",
            version_or_model="docker",
            role="Controlled relational fixture",
            participant_class=ParticipantClass.CONTROLLED_FIXTURE,
        ),
        ParticipantEvidence(
            participant_id="proof-evaluator",
            name="TOOLS SQL investigation evaluator",
            implementation="platform_proofs",
            version_or_model=PROOF_VERSION,
            role="Scenario invariant checks",
            participant_class=ParticipantClass.PROOF_OWNED,
        ),
    )


def _build_architecture(
    participants: tuple[ParticipantEvidence, ...],
) -> ArchitectureEvidence:
    return ArchitectureEvidence(
        participants=participants,
        edges=(
            ArchitectureEdgeEvidence(
                from_participant="llm-provider",
                to_participant="intergrax-runtime",
                relationship="native_tool_planning",
            ),
            ArchitectureEdgeEvidence(
                from_participant="intergrax-runtime",
                to_participant="sql-tool",
                relationship="bounded_tool_invocation",
            ),
            ArchitectureEdgeEvidence(
                from_participant="sql-tool",
                to_participant="postgres-fixture",
                relationship="read_only_sql",
            ),
            ArchitectureEdgeEvidence(
                from_participant="intergrax-runtime",
                to_participant="proof-evaluator",
                relationship="scenario_evaluation",
            ),
        ),
    )


def _build_claim() -> ProofClaimEvidence:
    return ProofClaimEvidence(
        claim=(
            "The bounded iterative tool runtime can use real SQL observations from "
            "PostgreSQL via a real LLM provider to drive subsequent evidence-dependent "
            "tool calls, preserve an explicit InvestigationProof chain, and reach a "
            "bounded conclusion while rejecting unsupported causal claims."
        ),
        user_relevance=(
            "Demonstrates that Intergrax can orchestrate real tool loops with auditable "
            "evidence chains for investigative SQL workloads."
        ),
        success_criteria=(
            "Minimum successful tool-call counts per scenario.",
            "Valid evidence-dependent follow-up chain (ENG-6).",
            "Scenario-specific semantic outcomes pass evaluator checks.",
            "Normal bounded termination stop_reason.",
        ),
        falsification_criteria=(
            "Missing or invalid evidence basis on follow-up tool rounds.",
            "Unsupported causal or staffing claims in final answers.",
            "Volume-only explanation accepted for scenario A.",
        ),
        excluded_claims=(
            "Production readiness or commercial validation.",
            "Universal provider compatibility.",
            "Real-user workflow validation.",
            "All deployment modes and workloads.",
        ),
    )


def _build_environment(result: ToolsSqlInvestigationProofResult) -> EnvironmentEvidence:
    identity = result.dataset_identity
    return EnvironmentEvidence(
        dataset=DatasetEnvironmentEvidence(
            dataset_id=str(identity.get("dataset_id", "")),
            dataset_version=str(identity.get("dataset_version", DATASET_VERSION)),
            row_count=int(identity.get("row_count", 0)),
            seed=int(identity["seed"]) if identity.get("seed") is not None else None,
            scenario_version=str(identity.get("ground_truth_version", "")),
            fingerprint_sha256=result.dataset_fingerprint_sha256,
            infrastructure_identity="docker-postgresql-fixture",
            access_mode="read_only_sql",
            ground_truth_checks=tuple(
                f"{key}={value}"
                for key, value in sorted(result.db_verification_stats.items())
            ),
            information_exposed_to_model=(
                "Investigation question only.",
                "SQL tool outputs bounded by MAX_VISIBLE_ROWS.",
                "No staffing columns in fixture schema.",
            ),
        )
    )


def _build_evaluator_checks(
    scenario: ScenarioRunResult,
    snapshot: ScenarioExecutionSnapshot,
) -> EvaluatorSummaryEvidence:
    trace_evidence_ids = tuple(
        _evidence_id_for_call(_tool_call_id(trace, index))
        for index, trace in enumerate(snapshot.tool_traces)
        if trace.success
    )
    final_evidence_id = f"evidence-final-answer-{scenario.scenario_id.value.lower()}"
    checks: list[EvaluatorCheckEvidence] = [
        EvaluatorCheckEvidence(
            check_id="minimum_tool_calls",
            label="Minimum successful tool-call count",
            passed=not any(
                reason.startswith("insufficient_tool_calls")
                for reason in scenario.failure_reasons
            ),
            explanation=proof_authored_report_safe_text(
                f"successful_tool_calls={scenario.successful_tool_calls}"
            ),
            evidence_ids=trace_evidence_ids[:1],
        ),
        EvaluatorCheckEvidence(
            check_id="investigation_proof_chain",
            label="Valid evidence-dependent follow-up chain",
            passed=snapshot.follow_up_has_valid_basis
            or scenario.investigation_proof_steps == 0,
            explanation=proof_authored_report_safe_text(
                f"investigation_proof_steps={scenario.investigation_proof_steps}"
            ),
            evidence_ids=trace_evidence_ids,
        ),
        EvaluatorCheckEvidence(
            check_id="termination",
            label="Normal bounded termination",
            passed=scenario.stop_reason == "planner_final_answer",
            explanation=proof_authored_report_safe_text(f"stop_reason={scenario.stop_reason}"),
            evidence_ids=(),
        ),
    ]
    if scenario.scenario_id is ScenarioId.A and scenario.outcome_a is not None:
        checks.append(
            EvaluatorCheckEvidence(
                check_id="scenario_a_semantics",
                label="Scenario A semantic outcome",
                passed=scenario.outcome_a.conclusion_supported,
                explanation=proof_authored_report_safe_text(
                    "North anomaly supported; volume-only rejected."
                ),
                evidence_ids=(final_evidence_id,),
            )
        )
    elif scenario.scenario_id is ScenarioId.B and scenario.outcome_b is not None:
        checks.append(
            EvaluatorCheckEvidence(
                check_id="scenario_b_semantics",
                label="Scenario B semantic outcome",
                passed=not scenario.outcome_b.claims_direct_causation
                and scenario.outcome_b.verifies_segmented_evidence,
                explanation=proof_authored_report_safe_text(
                    "Segmented evidence without direct causation claim."
                ),
                evidence_ids=(final_evidence_id,),
            )
        )
    elif scenario.scenario_id is ScenarioId.C and scenario.outcome_c is not None:
        checks.append(
            EvaluatorCheckEvidence(
                check_id="scenario_c_semantics",
                label="Scenario C semantic outcome",
                passed=scenario.outcome_c.reports_missing_evidence
                and not scenario.outcome_c.claims_staffing_cause,
                explanation=proof_authored_report_safe_text(
                    "Missing staffing evidence acknowledged."
                ),
                evidence_ids=(final_evidence_id,),
            )
        )
    return EvaluatorSummaryEvidence(
        passed=scenario.passed,
        checks=tuple(checks),
        failure_reasons=scenario.failure_reasons,
    )


def _scenario_status(
    scenario: ScenarioRunResult,
    global_status: ProofEvidenceExecutionStatus,
) -> ProofEvidenceExecutionStatus:
    if global_status in {
        ProofEvidenceExecutionStatus.BLOCKED,
        ProofEvidenceExecutionStatus.CRASH,
    }:
        return global_status
    return (
        ProofEvidenceExecutionStatus.PASS
        if scenario.passed
        else ProofEvidenceExecutionStatus.FAIL
    )


def _build_scenario_steps(
    snapshot: ScenarioExecutionSnapshot,
    *,
    scenario_id: ScenarioId,
) -> tuple[ProofExecutionStep, ...]:
    steps: list[ProofExecutionStep] = []
    proof = snapshot.investigation_proof
    basis_by_call: dict[str, tuple[str, ...]] = {}
    purpose_by_call: dict[str, str] = {}
    if proof is not None:
        for proof_step in proof.steps:
            for call_id in proof_step.next_tool_call_ids:
                basis_by_call[call_id] = proof_step.basis_tool_call_ids
                purpose_by_call[call_id] = proof_step.public_reason or "investigate"

    for index, trace in enumerate(snapshot.tool_traces):
        call_id = _tool_call_id(trace, index)
        evidence_id = _evidence_id_for_call(call_id)
        sql = ""
        if trace.arguments:
            raw_sql = trace.arguments.get("sql")
            if isinstance(raw_sql, str):
                sql = raw_sql
        basis_ids = tuple(
            _evidence_id_for_call(basis_id) for basis_id in basis_by_call.get(call_id, ())
        )
        purpose = purpose_by_call.get(call_id) or (
            f"Execute SQL tool call for scenario {scenario_id.value}"
        )
        steps.append(
            ProofExecutionStep(
                step_index=index,
                step_id=f"{scenario_id.value.lower()}-step-{index + 1}",
                purpose=proof_authored_report_safe_text(purpose),
                evidence_basis_ids=basis_ids,
                action=proof_authored_report_safe_text(f"Invoke {trace.tool_name}"),
                input=_safe_sql_arguments(sql) if sql else None,
                observation=_safe_output_preview(trace.output_preview or "")
                if trace.output_preview
                else None,
                evidence_created_ids=(evidence_id,),
                status=ProofStepExecutionStatus.OK
                if trace.success
                else ProofStepExecutionStatus.FAIL,
                participant_id="sql-tool",
                tool_invocation=ToolInvocationEvidence(
                    tool_id=trace.tool_name,
                    provider_tool_name=trace.tool_name,
                    call_id=call_id,
                    safe_arguments=_safe_sql_arguments(sql) if sql else None,
                    success=trace.success,
                    output_summary=explicit_runtime_report_safe_text(
                        (trace.output_preview or "")[:240]
                    ),
                    bounded_output=_safe_output_preview(trace.output_preview or "")
                    if trace.output_preview
                    else None,
                    error=(
                        sanitized_runtime_report_safe_text(trace.error_message)
                        if trace.error_message
                        else None
                    ),
                ),
                error=(
                    sanitized_runtime_report_safe_text(trace.error_message)
                    if trace.error_message
                    else None
                ),
            )
        )
    return tuple(steps)


def _build_scenario_graph(
    snapshot: ScenarioExecutionSnapshot,
    *,
    scenario_id: ScenarioId,
) -> EvidenceGraphEvidence:
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    node_ids: set[str] = set()
    for index, trace in enumerate(snapshot.tool_traces):
        call_id = _tool_call_id(trace, index)
        evidence_id = _evidence_id_for_call(call_id)
        step_id = f"{scenario_id.value.lower()}-step-{index + 1}"
        nodes.append(
            EvidenceNode(
                evidence_id=evidence_id,
                kind=EvidenceNodeKind.TOOL_RESULT,
                label=f"Tool result {call_id}",
                summary=explicit_runtime_report_safe_text(
                    (trace.output_preview or "")[:120]
                ),
                producing_step_id=step_id,
            )
        )
        node_ids.add(evidence_id)
        edges.append(
            EvidenceEdge(
                from_evidence_id=evidence_id,
                to_step_id=step_id,
                relationship=EvidenceRelationship.PRODUCED_BY,
            )
        )
    proof = snapshot.investigation_proof
    if proof is not None:
        for proof_step in proof.steps:
            for call_id in proof_step.next_tool_call_ids:
                target_id = _evidence_id_for_call(call_id)
                if target_id not in node_ids:
                    continue
                for basis_id in proof_step.basis_tool_call_ids:
                    basis_evidence = _evidence_id_for_call(basis_id)
                    if basis_evidence not in node_ids:
                        continue
                    edges.append(
                        EvidenceEdge(
                            from_evidence_id=basis_evidence,
                            to_evidence_id=target_id,
                            relationship=EvidenceRelationship.EVIDENCE_BASIS,
                        )
                    )
    if snapshot.final_answer.strip():
        final_id = f"evidence-final-answer-{scenario_id.value.lower()}"
        nodes.append(
            EvidenceNode(
                evidence_id=final_id,
                kind=EvidenceNodeKind.FINAL_ANSWER,
                label="Final answer",
                summary=explicit_runtime_report_safe_text(snapshot.final_answer[:120]),
            )
        )
        node_ids.add(final_id)
        for node in nodes:
            if node.kind == EvidenceNodeKind.TOOL_RESULT:
                edges.append(
                    EvidenceEdge(
                        from_evidence_id=node.evidence_id,
                        to_evidence_id=final_id,
                        relationship=EvidenceRelationship.SUPPORTS_CONCLUSION,
                    )
                )
    return EvidenceGraphEvidence(nodes=tuple(nodes), edges=tuple(edges))


def _build_scenario_evidence(
    scenario: ScenarioRunResult,
    snapshot: ScenarioExecutionSnapshot,
    *,
    global_status: ProofEvidenceExecutionStatus,
) -> ScenarioEvidence:
    scenario_def = _scenario_lookup()[scenario.scenario_id]
    expected, falsification = _SCENARIO_EXPECTATIONS[scenario.scenario_id]
    steps = _build_scenario_steps(snapshot, scenario_id=scenario.scenario_id)
    return ScenarioEvidence(
        scenario_id=scenario.scenario_id.value,
        title=f"Scenario {scenario.scenario_id.value}",
        question=scenario_def.question,
        expected_behavior=expected,
        falsification_condition=falsification,
        execution_status=_scenario_status(scenario, global_status),
        metrics=(
            MetricEvidence(
                name="successful_tool_calls",
                value=scenario.successful_tool_calls,
                unit="count",
            ),
            MetricEvidence(
                name="investigation_proof_steps",
                value=scenario.investigation_proof_steps,
                unit="count",
            ),
        ),
        steps=steps,
        final_output=FinalOutputEvidence(
            present=bool(snapshot.final_answer.strip()),
            output_type="text",
            content=explicit_runtime_report_safe_text(snapshot.final_answer),
            report_safe=True,
            evidence_basis_ids=tuple(
                _evidence_id_for_call(_tool_call_id(trace, index))
                for index, trace in enumerate(snapshot.tool_traces)
                if trace.success
            ),
        ),
        evaluator=_build_evaluator_checks(scenario, snapshot),
        failure=(
            FailureEvidence(
                classification=FailureClassification.MODEL_BEHAVIOR_FAILURE,
                message=sanitized_runtime_report_safe_text(
                    "; ".join(scenario.failure_reasons) or "scenario failed"
                ),
                evidence_ids=(f"evidence-final-answer-{scenario.scenario_id.value.lower()}",),
            )
            if not scenario.passed
            else None
        ),
    )


def _merge_graphs(
    snapshots: tuple[ScenarioExecutionSnapshot, ...],
    scenarios: tuple[ScenarioRunResult, ...],
) -> EvidenceGraphEvidence:
    nodes: list[EvidenceNode] = []
    edges: list[EvidenceEdge] = []
    for snapshot, scenario in zip(snapshots, scenarios, strict=False):
        graph = _build_scenario_graph(snapshot, scenario_id=scenario.scenario_id)
        nodes.extend(graph.nodes)
        edges.extend(graph.edges)
    return EvidenceGraphEvidence(nodes=tuple(nodes), edges=tuple(edges))


def _build_tools_extension(
    snapshots: tuple[ScenarioExecutionSnapshot, ...],
) -> ToolsSqlInvestigationExtension:
    sql_statements: list[str] = []
    observations: list[ToolsSqlObservationEvidence] = []
    total_calls = 0
    stop_reason = ""
    proof_steps = 0
    follow_up_basis: bool | None = None
    for snapshot in snapshots:
        stop_reason = snapshot.stop_reason or stop_reason
        proof_steps = max(proof_steps, snapshot.investigation_proof_steps)
        follow_up_basis = snapshot.follow_up_has_valid_basis
        for index, trace in enumerate(snapshot.tool_traces):
            sql = ""
            if trace.arguments:
                raw_sql = trace.arguments.get("sql")
                if isinstance(raw_sql, str):
                    sql = raw_sql
            if sql:
                sql_statements.append(sql)
            observations.append(
                ToolsSqlObservationEvidence(
                    call_index=index,
                    tool_id=trace.tool_name,
                    sql_text=sql,
                    output_preview=trace.output_preview or "",
                    success=trace.success,
                )
            )
            if trace.success:
                total_calls += 1
    return ToolsSqlInvestigationExtension(
        sql_statements=tuple(sql_statements),
        tool_observations=tuple(observations),
        investigation_proof_step_count=proof_steps,
        successful_tool_calls=total_calls,
        stop_reason=stop_reason,
        follow_up_has_valid_basis=follow_up_basis,
    )


def _blocked_failure(result: ToolsSqlInvestigationProofResult) -> FailureEvidence:
    reason = result.blocked_reason or "blocked"
    classification = FailureClassification.BLOCKED_CONFIGURATION
    if "provider" in reason.lower() or "credentials" in reason.lower():
        classification = FailureClassification.PROVIDER_CONFIGURATION
    return FailureEvidence(
        classification=classification,
        message=sanitized_runtime_report_safe_text(reason),
        completed_milestones=("dataset identity resolved",),
        failed_milestone="provider or runtime configuration",
        skipped_not_reached=(
            "adapter construction",
            "bounded tool loop",
            "scenario evaluation",
        ),
    )


def _build_conclusion(
    status: ProofEvidenceExecutionStatus,
    result: ToolsSqlInvestigationProofResult,
) -> ConclusionEvidence:
    if status == ProofEvidenceExecutionStatus.PASS:
        return ConclusionEvidence(
            supported_conclusions=(
                "Bounded iterative SQL investigation completed with evaluator PASS.",
            ),
            unsupported_conclusions=(
                "No claim of production readiness or universal provider support.",
            ),
        )
    if status == ProofEvidenceExecutionStatus.BLOCKED:
        return ConclusionEvidence(
            unsupported_conclusions=("Execution blocked before live scenarios.",),
            open_questions=(result.blocked_reason or "blocked",),
        )
    return ConclusionEvidence(
        unsupported_conclusions=("Claim not demonstrated under named proof conditions.",),
        open_questions=tuple(
            reason
            for scenario in result.scenarios
            for reason in scenario.failure_reasons
        ),
    )


def build_tools_sql_investigation_evidence(
    context: ToolsSqlInvestigationEvidenceBuildContext,
) -> PlatformProofEvidence:
    result = context.proof_result
    status = _resolve_execution_status(result, context.execution_status)
    finished_at = context.finished_at
    execution_id = context.execution_id or f"{PROOF_ID}-{int(context.started_at.timestamp())}"
    generated_at = finished_at or datetime.now(UTC)
    participants = _build_participants(result.model_provider)
    scenario_items = tuple(
        _build_scenario_evidence(scenario, snapshot, global_status=status)
        for scenario, snapshot in zip(result.scenarios, context.scenario_snapshots, strict=False)
    )
    failure = context.failure
    if failure is None and status == ProofEvidenceExecutionStatus.BLOCKED:
        failure = _blocked_failure(result)
    graph = _merge_graphs(context.scenario_snapshots, result.scenarios)
    final_answer = ""
    if context.scenario_snapshots:
        final_answer = context.scenario_snapshots[-1].final_answer
    evidence = PlatformProofEvidence(
        proof_identity=ProofIdentityEvidence(
            proof_id=result.proof_id,
            title=PROOF_TITLE,
            domain=PROOF_DOMAIN,
            proof_version=PROOF_VERSION,
            source_revision=context.source_revision,
            execution_profile=context.execution_profile,
        ),
        execution=ExecutionMetadataEvidence(
            status=status,
            started_at=context.started_at,
            finished_at=finished_at,
            duration_ms=_duration_ms(context.started_at, finished_at),
            platform=context.platform,
            runtime_version=context.runtime_version,
            source_dirty=context.source_dirty,
            command_executable="uv",
            command_argv=("run", "python", "platform_proofs/tools/iterative_sql_investigation/run_proof.py"),
        ),
        claim=_build_claim(),
        architecture=_build_architecture(participants),
        participants=participants,
        environment=_build_environment(result),
        scenarios=scenario_items,
        evidence_graph=graph,
        final_output=FinalOutputEvidence(
            present=bool(final_answer.strip()),
            output_type="text",
            content=explicit_runtime_report_safe_text(final_answer),
            report_safe=True,
        )
        if final_answer.strip()
        else None,
        evaluator=EvaluatorSummaryEvidence(
            passed=result.overall_pass,
            checks=tuple(
                check
                for scenario in scenario_items
                if scenario.evaluator is not None
                for check in scenario.evaluator.checks
            ),
            failure_reasons=tuple(
                reason
                for scenario in result.scenarios
                for reason in scenario.failure_reasons
            ),
        )
        if result.scenarios
        else None,
        limitations=(
            "Single configured LLM provider/model per run.",
            "Fixture dataset with bounded SQL row visibility.",
            "Docker-hosted PostgreSQL fixture required.",
        ),
        conclusion=_build_conclusion(status, result),
        reproduction=ReproductionEvidence(
            source_revision=context.source_revision,
            command=context.command
            or "uv run python platform_proofs/tools/iterative_sql_investigation/run_proof.py",
            prerequisites=("uv", "docker", "INTERGRAX_LLM_PROVIDER"),
            required_env_variable_names=context.required_env_variable_names,
            dataset_fingerprint_sha256=result.dataset_fingerprint_sha256,
        ),
        provenance=ProvenanceEvidence(
            proof_id=result.proof_id,
            source_revision=context.source_revision,
            generated_at=generated_at,
            execution_id=execution_id,
            artifact_identity=build_artifact_identity(
                proof_id=result.proof_id,
                execution_id=execution_id,
                generated_at=generated_at,
            ),
        ),
        failure=failure,
        domain_extension=DomainExtensionEvidence(
            tools=_build_tools_extension(context.scenario_snapshots)
            if context.scenario_snapshots
            else None
        ),
    )
    return evidence
