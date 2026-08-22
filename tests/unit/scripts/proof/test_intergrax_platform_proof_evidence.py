# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.investigation_proof import InvestigationProof, InvestigationProofStep
from platform_proofs.tools.iterative_sql_investigation.contracts import (
    PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
    SqlQueryInput,
)
from platform_proofs.tools.iterative_sql_investigation.dataset_identity import (
    DatasetIdentity,
    compute_dataset_fingerprint,
)
from platform_proofs.tools.iterative_sql_investigation.evaluator import (
    build_execution_snapshot,
    evaluate_scenario,
)
from platform_proofs.tools.iterative_sql_investigation.evidence_builder import (
    ToolsSqlInvestigationEvidenceBuildContext,
    build_tools_sql_investigation_evidence,
)
from platform_proofs.tools.iterative_sql_investigation.proof_result import (
    ModelProviderIdentity,
    ToolsSqlInvestigationProofResult,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ScenarioId
from scripts.proof.intergrax_platform_proof_evidence import (
    PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION,
    ArchitectureEvidence,
    EnvironmentEvidence,
    EvidenceEdge,
    EvidenceGraphEvidence,
    EvidenceNode,
    EvidenceNodeKind,
    EvidenceRelationship,
    ExecutionMetadataEvidence,
    FailureClassification,
    FailureEvidence,
    FinalOutputEvidence,
    PlatformProofEvidence,
    ProofClaimEvidence,
    ProofEvidenceExecutionStatus,
    ProofIdentityEvidence,
    ProvenanceEvidence,
    ReportSafeField,
    ReportSafePayload,
    ReportSafeText,
    ReportSafeTextSourceKind,
    ReportSafeVisibility,
    ReproductionEvidence,
    ToolInvocationEvidence,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    redacted_report_safe_text,
    sanitized_runtime_report_safe_text,
    sanitize_untrusted_report_text,
    execution_status_from_proof_status,
)
from scripts.proof.intergrax_platform_proof_evidence_io import (
    EVIDENCE_FILENAME,
    serialize_evidence_deterministic as io_serialize,
    with_evidence_checksum,
    write_evidence_json,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile, ProofStatus, SuiteReceipt

pytestmark = pytest.mark.unit


def _sql_traces(*sql_queries: str) -> tuple[ToolCallTrace, ...]:
    return tuple(
        ToolCallTrace(
            tool_name=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
            arguments=SqlQueryInput(sql=sql).model_dump(),
            output_preview="North express long_haul rate 0.68",
            success=True,
            error_message=None,
            raw_trace={"tool_call_id": f"tc-{index + 1}"},
        )
        for index, sql in enumerate(sql_queries)
    )


def _pass_snapshot():
    traces = _sql_traces(
        "SELECT region, AVG(delayed::int) FROM proof.parcel_events GROUP BY region",
        "SELECT origin_hub, AVG(delayed::int) FROM proof.parcel_events GROUP BY origin_hub",
        (
            "SELECT service_type, route_type, AVG(delayed::int) FROM proof.parcel_events "
            "WHERE region='North' GROUP BY service_type, route_type"
        ),
    )
    proof = InvestigationProof(
        steps=(
            InvestigationProofStep(
                round_index=1,
                basis_tool_call_ids=(),
                next_tool_call_ids=("tc-1",),
                public_reason="compare regions",
            ),
            InvestigationProofStep(
                round_index=2,
                basis_tool_call_ids=("tc-1",),
                next_tool_call_ids=("tc-2",),
                public_reason="inspect segment",
            ),
            InvestigationProofStep(
                round_index=3,
                basis_tool_call_ids=("tc-1", "tc-2"),
                next_tool_call_ids=("tc-3",),
                public_reason="hub rates",
            ),
        ),
        final_available_evidence_ids=("tc-1", "tc-2", "tc-3"),
    )
    return build_execution_snapshot(
        traces=traces,
        investigation_proof=proof,
        stop_reason="planner_final_answer",
        final_answer=(
            "North delays are driven by the North express long_haul segment; "
            "normalized hub rates falsify a volume-only explanation."
        ),
    )


def _base_result(*, overall_pass: bool, scenarios: tuple = (), blocked_reason: str | None = None):
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    return ToolsSqlInvestigationProofResult(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        dataset_identity=identity.as_dict(),
        dataset_fingerprint_sha256=fingerprint.sha256,
        db_verification_stats={"total_rows": identity.row_count},
        model_provider=ModelProviderIdentity(
            provider="openai",
            model="gpt-test",
            supports_native_tools=True,
        ),
        scenarios=scenarios,
        overall_pass=overall_pass,
        blocked_reason=blocked_reason,
    )


def _build_context(
    result: ToolsSqlInvestigationProofResult,
    *,
    snapshots: tuple = (),
    status: ProofEvidenceExecutionStatus | None = None,
    failure: FailureEvidence | None = None,
) -> ToolsSqlInvestigationEvidenceBuildContext:
    started = datetime(2026, 8, 21, 12, 0, 0, tzinfo=UTC)
    finished = datetime(2026, 8, 21, 12, 5, 0, tzinfo=UTC)
    return ToolsSqlInvestigationEvidenceBuildContext(
        proof_result=result,
        scenario_snapshots=snapshots,
        started_at=started,
        finished_at=finished,
        source_revision="abc123",
        source_dirty=False,
        execution_profile=ProofProfile.FULL,
        platform="linux",
        runtime_version="3.12.0",
        execution_id="run-test-1",
        execution_status=status,
        failure=failure,
    )


def test_pass_evidence_serializes() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    result = _base_result(overall_pass=True, scenarios=(scenario,))
    evidence = build_tools_sql_investigation_evidence(
        _build_context(result, snapshots=(snapshot,))
    )
    payload = json.loads(io_serialize(evidence))
    assert payload["schema_version"] == PLATFORM_PROOF_EVIDENCE_SCHEMA_VERSION
    assert payload["execution"]["status"] == "PASS"
    assert payload["domain_extension"]["tools"]["extension_id"] == "tools.tool-call-trace"


def test_fail_evidence_serializes() -> None:
    traces = _sql_traces(
        "SELECT weight_kg, AVG(delayed::int) FROM proof.parcel_events GROUP BY weight_kg",
        "SELECT route_type, service_type, weight_kg, delayed FROM proof.parcel_events",
    )
    snapshot = build_execution_snapshot(
        traces=traces,
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer="Heavier weight causes delays across the network.",
    )
    scenario = evaluate_scenario(ScenarioId.B, snapshot)
    result = _base_result(overall_pass=False, scenarios=(scenario,))
    evidence = build_tools_sql_investigation_evidence(
        _build_context(result, snapshots=(snapshot,))
    )
    assert evidence.execution.status is ProofEvidenceExecutionStatus.FAIL


def test_blocked_evidence_serializes() -> None:
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    result = ToolsSqlInvestigationProofResult.blocked(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        identity=identity,
        fingerprint=fingerprint,
        reason="missing required proof configuration",
    )
    evidence = build_tools_sql_investigation_evidence(_build_context(result))
    assert evidence.execution.status is ProofEvidenceExecutionStatus.BLOCKED
    assert evidence.failure is not None
    assert evidence.failure.classification is FailureClassification.BLOCKED_CONFIGURATION


def test_crash_partial_evidence_serializes() -> None:
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    result = ToolsSqlInvestigationProofResult(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        dataset_identity=identity.as_dict(),
        dataset_fingerprint_sha256=fingerprint.sha256,
        db_verification_stats={"total_rows": identity.row_count},
        model_provider=ModelProviderIdentity(
            provider="openai",
            model="gpt-test",
            supports_native_tools=True,
        ),
        scenarios=(),
        overall_pass=False,
    )
    failure = FailureEvidence(
        classification=FailureClassification.UNKNOWN,
        message=sanitized_runtime_report_safe_text("OpenAI request failed"),
        completed_milestones=("dataset verified", "adapter constructed"),
        failed_milestone="OpenAI request",
        skipped_not_reached=("model function call", "SQL execution", "scenarios B/C"),
    )
    evidence = build_tools_sql_investigation_evidence(
        _build_context(
            result,
            status=ProofEvidenceExecutionStatus.CRASH,
            failure=failure,
        )
    )
    payload = json.loads(io_serialize(evidence))
    assert payload["execution"]["status"] == "CRASH"
    assert payload["scenarios"] == []
    assert payload["failure"]["failed_milestone"] == "OpenAI request"
    assert payload["failure"]["message"]["text"] == "OpenAI request failed"


def test_execution_step_preserves_operational_fields() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    step = evidence.scenarios[0].steps[1]
    assert step.purpose.text == "inspect segment"
    assert step.evidence_basis_ids == ("evidence-a-tc-1",)
    assert PLATFORM_PROOF_SQL_QUERY_TOOL_ID in step.action.text
    assert step.observation is not None
    assert step.evidence_created_ids == ("evidence-a-tc-2",)


def test_tool_invocation_preserves_canonical_tool_id() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    invocation = evidence.scenarios[0].steps[0].tool_invocation
    assert invocation is not None
    assert invocation.tool_id == PLATFORM_PROOF_SQL_QUERY_TOOL_ID


def test_report_safe_payload_rejects_secret_field() -> None:
    with pytest.raises(ValueError, match="secret-bearing field"):
        ReportSafePayload(
            fields=(
                ReportSafeField(
                    name="api_key",
                    visibility=ReportSafeVisibility.REPORT_SAFE,
                    value=proof_authored_report_safe_text("super-secret"),
                ),
            )
        )


def test_report_safe_field_rejects_raw_runtime_string() -> None:
    with pytest.raises(ValueError, match="raw runtime string"):
        ReportSafeField(
            name="output",
            visibility=ReportSafeVisibility.REPORT_SAFE,
            value="Authorization: Bearer fake-secret-value",
        )


def test_report_safe_text_serializes() -> None:
    safe = explicit_runtime_report_safe_text("North express long_haul rate 0.68")
    payload = safe.model_dump(mode="json")
    assert payload["text"] == "North express long_haul rate 0.68"
    assert payload["visibility"] == "REPORT_SAFE"
    assert payload["source_kind"] == "RUNTIME_EXPLICIT"


def test_sanitize_bearer_header() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "Authorization: Bearer fake-secret-value"
    )
    assert redaction_applied is True
    assert "fake-secret-value" not in sanitized
    assert "Bearer [REDACTED]" in sanitized


def test_sanitize_env_assignment() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "missing config OPENAI_API_KEY=fake-secret"
    )
    assert redaction_applied is True
    assert "fake-secret" not in sanitized
    assert "OPENAI_API_KEY=[REDACTED]" in sanitized


def test_sanitize_credential_url() -> None:
    sanitized, redaction_applied = sanitize_untrusted_report_text(
        "connect postgresql://proof_user:proof_pass@localhost:5432/proof_db"
    )
    assert redaction_applied is True
    assert "proof_pass" not in sanitized
    assert "[REDACTED]:[REDACTED]@" in sanitized


def test_ordinary_sql_unchanged() -> None:
    sql = (
        "SELECT service_type, route_type, AVG(delayed::int) "
        "FROM proof.parcel_events WHERE region='North' "
        "GROUP BY service_type, route_type"
    )
    safe = explicit_runtime_report_safe_text(sql)
    assert safe.text == sql
    assert safe.redaction_applied is False


def test_ordinary_sql_output_preview_unchanged() -> None:
    preview = "North express long_haul rate 0.68"
    safe = explicit_runtime_report_safe_text(preview)
    assert safe.text == preview
    assert safe.redaction_applied is False


def test_provider_error_without_secret_remains_readable() -> None:
    message = "400 Invalid tools[0].name"
    safe = sanitized_runtime_report_safe_text(message)
    assert safe.text == message
    assert safe.redaction_applied is False


def test_provider_error_with_fake_token_redacted_in_serialization() -> None:
    failure = FailureEvidence(
        classification=FailureClassification.PROVIDER_CONFIGURATION,
        message=sanitized_runtime_report_safe_text(
            "provider rejected request Authorization: Bearer fake-secret-value"
        ),
    )
    payload = failure.model_dump(mode="json")
    serialized = json.dumps(payload)
    assert "fake-secret-value" not in serialized
    assert "Bearer [REDACTED]" in payload["message"]["text"]


def test_final_model_answer_explicitly_report_safe() -> None:
    answer = "North delays are driven by the North express long_haul segment."
    final_output = FinalOutputEvidence(
        present=True,
        content=explicit_runtime_report_safe_text(answer),
        report_safe=True,
    )
    assert final_output.content.source_kind is ReportSafeTextSourceKind.RUNTIME_EXPLICIT
    assert final_output.content.text == answer


def test_redacted_text_does_not_expose_original_value() -> None:
    redacted = redacted_report_safe_text()
    payload = redacted.model_dump(mode="json")
    assert payload["visibility"] == "REDACTED"
    assert payload["text"] == "[REDACTED]"
    assert payload["redaction_applied"] is True


def test_tool_invocation_rejects_raw_runtime_summary() -> None:
    with pytest.raises(ValueError, match="raw runtime string"):
        ToolInvocationEvidence(
            tool_id="sql-query",
            success=True,
            output_summary="Authorization: Bearer fake-secret-value",
        )


def test_failure_evidence_builder_redacts_provider_secret() -> None:
    traces = (
        ToolCallTrace(
            tool_name=PLATFORM_PROOF_SQL_QUERY_TOOL_ID,
            arguments=SqlQueryInput(sql="SELECT 1").model_dump(),
            output_preview="",
            success=False,
            error_message="Authorization: Bearer fake-secret-value",
            raw_trace={"tool_call_id": "tc-err"},
        ),
    )
    snapshot = build_execution_snapshot(
        traces=traces,
        investigation_proof=None,
        stop_reason="provider_error",
        final_answer="",
    )
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=False, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    serialized = io_serialize(evidence)
    assert "fake-secret-value" not in serialized
    step = evidence.scenarios[0].steps[0]
    assert step.tool_invocation is not None
    assert step.tool_invocation.error is not None
    assert "Bearer [REDACTED]" in step.tool_invocation.error.text


def test_evaluator_checks_reference_graph_evidence_ids() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    graph_ids = {node.evidence_id for node in evidence.evidence_graph.nodes}
    evaluator = evidence.scenarios[0].evaluator
    assert evaluator is not None
    for check in evaluator.checks:
        for evidence_id in check.evidence_ids:
            assert evidence_id in graph_ids or evidence_id.startswith("evidence-")


def test_evidence_graph_rejects_dangling_references() -> None:
    with pytest.raises(ValueError, match="dangling from_evidence_id"):
        PlatformProofEvidence(
            proof_identity=ProofIdentityEvidence(
                proof_id="P",
                title="t",
                domain="tools",
                proof_version="v1",
                source_revision="sha",
                execution_profile=ProofProfile.QUICK,
            ),
            execution=ExecutionMetadataEvidence(
                status=ProofEvidenceExecutionStatus.PASS,
                started_at=datetime(2026, 1, 1, tzinfo=UTC),
                finished_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
                platform="linux",
            ),
            claim=ProofClaimEvidence(
                claim="c",
                user_relevance="u",
                success_criteria=("s",),
                falsification_criteria=("f",),
                excluded_claims=("e",),
            ),
            architecture=_minimal_architecture(),
            participants=_minimal_architecture().participants,
            environment=_minimal_environment(),
            evidence_graph=EvidenceGraphEvidence(
                nodes=(
                    EvidenceNode(
                        evidence_id="evidence-a",
                        kind=EvidenceNodeKind.OTHER,
                        label="a",
                    ),
                ),
                edges=(
                    EvidenceEdge(
                        from_evidence_id="missing",
                        to_evidence_id="evidence-a",
                        relationship=EvidenceRelationship.EVIDENCE_BASIS,
                    ),
                ),
            ),
            reproduction=_minimal_reproduction(),
            provenance=_minimal_provenance(),
        )


def test_tools_domain_extension_serializes() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    extension = evidence.domain_extension.tools
    assert extension is not None
    assert extension.successful_tool_calls == 3
    assert extension.stop_reason == "planner_final_answer"
    assert len(extension.sql_statements) == 3


def test_deterministic_serialization() -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    first = io_serialize(with_evidence_checksum(evidence))
    second = io_serialize(with_evidence_checksum(evidence))
    assert first == second


def test_suite_receipt_contract_unchanged() -> None:
    assert SuiteReceipt.model_fields["schema_version"].default == "intergrax.proof_suite_receipt.v1"


def test_tools_proof_result_contract_unchanged() -> None:
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    blocked = ToolsSqlInvestigationProofResult.blocked(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        identity=identity,
        fingerprint=fingerprint,
        reason="test",
    )
    assert blocked.overall_pass is False
    assert blocked.blocked_reason == "test"


def test_no_html_renderer_module_present() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    assert not (repo_root / "scripts" / "proof" / "platform_proof_report_renderer.py").exists()


def test_write_evidence_json_to_run_directory(tmp_path: Path) -> None:
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    evidence = build_tools_sql_investigation_evidence(
        _build_context(_base_result(overall_pass=True, scenarios=(scenario,)), snapshots=(snapshot,))
    )
    run_directory = tmp_path / "run-1"
    path = write_evidence_json(evidence, proof_directory=run_directory)
    assert path == run_directory / EVIDENCE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["provenance"]["evidence_checksum"]


def test_execution_status_mapping_from_proof_status() -> None:
    assert execution_status_from_proof_status(ProofStatus.PASS) is ProofEvidenceExecutionStatus.PASS
    assert (
        execution_status_from_proof_status(ProofStatus.BLOCKED_ENVIRONMENT)
        is ProofEvidenceExecutionStatus.BLOCKED
    )


def _minimal_architecture() -> ArchitectureEvidence:
    from scripts.proof.intergrax_platform_proof_evidence import (
        ParticipantClass,
        ParticipantEvidence,
    )

    participant = ParticipantEvidence(
        participant_id="p1",
        name="n",
        implementation="i",
        version_or_model="v",
        role="r",
        participant_class=ParticipantClass.PLATFORM,
    )
    return ArchitectureEvidence(participants=(participant,))


def _minimal_environment() -> EnvironmentEvidence:
    return EnvironmentEvidence()


def _minimal_reproduction() -> ReproductionEvidence:
    return ReproductionEvidence(
        source_revision="sha",
        command="uv run python run_proof.py",
    )


def _minimal_provenance():
    return ProvenanceEvidence(
        proof_id="P",
        source_revision="sha",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        execution_id="exec-1",
        artifact_identity="artifact-1",
    )
