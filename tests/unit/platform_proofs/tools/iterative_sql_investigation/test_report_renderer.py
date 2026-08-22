# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import inspect
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import pytest

from intergrax.runtime.nexus.engine.runtime_state import ToolCallTrace
from intergrax.runtime.nexus.tools.investigation_proof import InvestigationProof, InvestigationProofStep
from platform_proofs.tools.iterative_sql_investigation.artifacts import (
    write_evidence,
    write_report,
)
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
from platform_proofs.tools.iterative_sql_investigation.report_renderer import (
    render_tools_sql_investigation_report,
    write_tools_sql_investigation_report,
)
from platform_proofs.tools.iterative_sql_investigation.scenarios import ScenarioId
from scripts.proof.intergrax_platform_proof_evidence import (
    FailureClassification,
    FailureEvidence,
    ProofEvidenceExecutionStatus,
    ReportSafeField,
    ReportSafePayload,
    ReportSafeVisibility,
    explicit_runtime_report_safe_text,
    proof_authored_report_safe_text,
    redacted_report_safe_text,
    sanitized_runtime_report_safe_text,
)
from scripts.proof.intergrax_platform_proof_evidence_io import EVIDENCE_FILENAME
from scripts.proof.intergrax_platform_proof_html_renderer import (
    REPORT_FILENAME,
    assert_no_external_report_dependencies,
    render_platform_proof_report,
)
from scripts.proof.intergrax_proof_contracts import ProofProfile

pytestmark = pytest.mark.unit

_INJECTION = "<script>alert(1)</script>"
_RAW_EVIDENCE_DUMP_RE = re.compile(r'"proof_identity"\s*:\s*\{')
_CHAIN_OF_THOUGHT_RE = re.compile(
    r"chain[- ]of[- ]thought|private_reasoning|scratchpad",
    re.IGNORECASE,
)


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
        source_revision="abc123def456",
        source_dirty=False,
        execution_profile=ProofProfile.FULL,
        platform="linux",
        runtime_version="3.12.0",
        execution_id="run-test-tools-report",
        execution_status=status,
        failure=failure,
    )


def _pass_evidence():
    snapshot = _pass_snapshot()
    scenario = evaluate_scenario(ScenarioId.A, snapshot)
    result = _base_result(overall_pass=True, scenarios=(scenario,))
    return build_tools_sql_investigation_evidence(
        _build_context(result, snapshots=(snapshot,))
    )


def _fail_evidence():
    traces = _sql_traces(
        "SELECT weight_kg, AVG(delayed::int) FROM proof.parcel_events GROUP BY weight_kg",
    )
    snapshot = build_execution_snapshot(
        traces=traces,
        investigation_proof=None,
        stop_reason="planner_final_answer",
        final_answer="Heavier weight causes delays across the network.",
    )
    scenario = evaluate_scenario(ScenarioId.B, snapshot)
    result = _base_result(overall_pass=False, scenarios=(scenario,))
    return build_tools_sql_investigation_evidence(
        _build_context(result, snapshots=(snapshot,))
    )


def _blocked_evidence():
    identity = DatasetIdentity.canonical()
    fingerprint = compute_dataset_fingerprint(identity)
    result = ToolsSqlInvestigationProofResult.blocked(
        proof_id="TOOLS-ITERATIVE-SQL-INVESTIGATION",
        identity=identity,
        fingerprint=fingerprint,
        reason="missing required proof configuration",
    )
    return build_tools_sql_investigation_evidence(_build_context(result))


def _crash_evidence():
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
        completed_milestones=("dataset verified",),
        failed_milestone="OpenAI request",
    )
    return build_tools_sql_investigation_evidence(
        _build_context(result, status=ProofEvidenceExecutionStatus.CRASH, failure=failure)
    )


def test_tools_renderer_uses_generic_template() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert html.startswith("<!DOCTYPE html>")
    assert "Report identity" in html
    assert "Executive summary" in html
    assert "Provenance" in html
    assert "tools-investigation-overview" in html


def test_generic_renderer_remains_tools_neutral() -> None:
    import scripts.proof.intergrax_platform_proof_html_renderer as renderer_module

    source = inspect.getsource(renderer_module)
    assert "ToolsSqlInvestigationExtension" not in source
    assert "iterative_sql_investigation" not in source
    assert 'proof_id ==' not in source


def test_pass_evidence_generates_nonempty_report() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert len(html) > 2000


def test_report_identity_matches_evidence() -> None:
    evidence = _pass_evidence()
    html = render_tools_sql_investigation_report(evidence)
    assert evidence.proof_identity.proof_id in html
    assert evidence.provenance.execution_id in html
    assert evidence.proof_identity.source_revision in html
    assert "PASS" in html


def test_domain_sections_render() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    for section_id in (
        "tools-investigation-overview",
        "tools-execution-topology",
        "tools-investigation-timeline",
        "tools-evidence-dependency",
        "tools-tool-call-detail",
        "tools-scenario-visualizations",
        "tools-evaluator-falsification",
        "tools-ground-truth-separation",
    ):
        assert f'id="{section_id}"' in html


def test_tool_call_trace_and_operational_fields_visible() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert "Purpose:" in html
    assert "Evidence basis:" in html
    assert "Action:" in html
    assert "Observation:" in html
    assert "compare regions" in html
    assert "inspect segment" in html
    assert "evidence-tc-1" in html
    assert PLATFORM_PROOF_SQL_QUERY_TOOL_ID in html
    assert "uses" in html


def test_dependency_chain_and_topology_visible() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert "Evidence dependency graph" in html or "tools-evidence-dependency" in html
    assert "observation → next action" in html
    assert "EVIDENCE_BASIS" in html or "evidence_basis" in html or "evidence-tc" in html


def test_scenario_abc_visualizations_visible() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert "Scenario A" in html
    assert "progressive narrowing" in html or "Region delay comparison" in html
    assert "Scenario outcome visualization" in html


def test_evaluator_separate_from_model_output() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert "Model / proof final output" in html
    assert "Evaluator verdict" in html
    assert "tools-evaluator-falsification" in html
    assert "This section is not the evaluator verdict" in html
    assert "WHY THIS PROOF PASSED" in html


def test_ground_truth_separation() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert "Ground truth known to proof" in html
    assert "Information available to model" in html
    assert "Investigation question only" in html
    assert "anomaly" not in html.lower().split("information available to model")[1].split("</section>")[0]


def test_redacted_and_injection_safe() -> None:
    evidence = _pass_evidence()
    step = evidence.scenarios[0].steps[0]
    bad_input = ReportSafePayload(
        summary=proof_authored_report_safe_text("test"),
        fields=(
            ReportSafeField(
                name="note",
                visibility=ReportSafeVisibility.REDACTED,
                value=redacted_report_safe_text(),
            ),
        ),
    )
    bad_step = step.model_copy(update={"input": bad_input})
    scenario = evidence.scenarios[0].model_copy(update={"steps": (bad_step,)})
    bad_evidence = evidence.model_copy(update={"scenarios": (scenario,)})
    html = render_tools_sql_investigation_report(bad_evidence)
    assert "[REDACTED]" in html
    assert "super-secret-token-value" not in html

    inj_evidence = evidence.model_copy(
        update={
            "scenarios": (
                evidence.scenarios[0].model_copy(
                    update={
                        "question": _INJECTION,
                    }
                ),
            )
        }
    )
    inj_html = render_tools_sql_investigation_report(inj_evidence)
    assert _INJECTION not in inj_html
    assert "&lt;script&gt;" in inj_html


def test_no_external_deps_or_raw_json() -> None:
    html = render_tools_sql_investigation_report(_pass_evidence())
    assert_no_external_report_dependencies(html)
    assert "<script" not in html.lower()
    assert not _RAW_EVIDENCE_DUMP_RE.search(html)
    assert not _CHAIN_OF_THOUGHT_RE.search(html)


def test_deterministic_output() -> None:
    evidence = _pass_evidence()
    assert render_tools_sql_investigation_report(evidence) == render_tools_sql_investigation_report(
        evidence
    )


def test_fail_blocked_crash_reports_render() -> None:
    for builder in (_fail_evidence, _blocked_evidence, _crash_evidence):
        html = render_tools_sql_investigation_report(builder())
        assert "Report identity" in html
        assert "tools-investigation-overview" in html


def test_artifacts_write_report_alongside_evidence(tmp_path: Path) -> None:
    evidence = _pass_evidence()
    write_evidence(evidence, run_directory=tmp_path)
    report_path = write_report(evidence, run_directory=tmp_path)
    assert (tmp_path / EVIDENCE_FILENAME).is_file()
    assert report_path == tmp_path / REPORT_FILENAME
    assert report_path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_missing_report_path_requires_directory() -> None:
    with pytest.raises(ValueError, match="output_path or run_directory"):
        write_tools_sql_investigation_report(_pass_evidence())


def test_preview_generation() -> None:
    preview_dir = Path(".artifacts/proof/report-preview/tools")
    preview_path = preview_dir / REPORT_FILENAME
    write_tools_sql_investigation_report(_pass_evidence(), run_directory=preview_dir)
    assert preview_path.is_file()
    content = preview_path.read_text(encoding="utf-8")
    assert "Iterative investigation timeline" in content
    assert "Tool execution topology" in content


def test_generic_without_domain_sections_unchanged_for_tools_evidence() -> None:
    evidence = _pass_evidence()
    generic = render_platform_proof_report(evidence)
    assert "Specialized presentation is not installed" in generic
    assert "tools-investigation-overview" not in generic
