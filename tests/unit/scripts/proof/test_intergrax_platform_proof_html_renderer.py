# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
from pathlib import Path

import pytest

from scripts.proof.intergrax_platform_proof_evidence import proof_authored_report_safe_text
from scripts.proof.intergrax_platform_proof_html_renderer import (
    PLATFORM_PROOF_HTML_RENDERER_VERSION,
    PLATFORM_PROOF_REPORT_SCHEMA_VERSION,
    PlatformProofReportRenderError,
    assert_no_external_report_dependencies,
    render_platform_proof_report,
    write_platform_proof_report,
)
from tests.unit.scripts.proof.intergrax_platform_proof_html_renderer_fixtures import (
    build_blocked_evidence,
    build_crash_evidence,
    build_fail_evidence,
    build_injection_evidence,
    build_multi_scenario_evidence,
    build_pass_evidence,
    build_redacted_payload_evidence,
)

pytestmark = pytest.mark.unit

_INJECTION = "<script>alert(1)</script>"
_RAW_EVIDENCE_DUMP_RE = re.compile(r'"proof_identity"\s*:\s*\{')
_CHAIN_OF_THOUGHT_RE = re.compile(
    r"chain[- ]of[- ]thought|private_reasoning|scratchpad",
    re.IGNORECASE,
)


def _render_pass() -> str:
    return render_platform_proof_report(build_pass_evidence())


def test_pass_evidence_renders_valid_html() -> None:
    html = _render_pass()
    assert html.startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert len(html) > 500


def test_fail_blocked_crash_evidence_render() -> None:
    for builder in (build_fail_evidence, build_blocked_evidence, build_crash_evidence):
        html = render_platform_proof_report(builder())
        assert "Report identity" in html
        assert "</html>" in html


def test_report_contains_core_identity_fields() -> None:
    html = _render_pass()
    assert "GENERIC-PLATFORM-PROOF" in html
    assert "PASS" in html
    assert "Platform can drive evidence-dependent tool calls" in html
    assert "Production readiness" in html


def test_participants_and_architecture_render() -> None:
    html = _render_pass()
    assert "LLM Provider" in html
    assert "PostgreSQL" in html
    assert "tool planning" in html
    assert "EXTERNAL_VENDOR" in html


def test_trace_steps_render_in_order() -> None:
    html = _render_pass()
    assert "Prepare deterministic dataset" in html
    assert "Inspect regional segment" in html
    assert "Execute bounded SQL query" in html
    assert "North region delay rate elevated" in html
    assert "evidence-dataset" in html
    assert "evidence-query-1" in html
    first_index = html.index("Prepare deterministic dataset")
    second_index = html.index("Inspect regional segment")
    assert first_index < second_index


def test_evaluator_separate_from_final_output() -> None:
    html = _render_pass()
    final_index = html.index("Model / proof final output")
    verdict_index = html.index("Evaluator verdict")
    assert final_index < verdict_index
    assert "North delays driven by express long_haul segment." in html
    assert "Scenario A passed" in html


def test_failure_boundary_renders_for_fail() -> None:
    html = render_platform_proof_report(build_fail_evidence())
    assert "MODEL_BEHAVIOR_FAILURE" in html
    assert "Failed milestone:</strong> evaluator</p>" in html


def test_limitations_reproduction_provenance_render() -> None:
    html = _render_pass()
    assert "Single provider run" in html
    assert "uv run python -m scripts.proof.intergrax_proof_runner" in html
    assert "abc123def456" in html
    assert PLATFORM_PROOF_REPORT_SCHEMA_VERSION in html
    assert PLATFORM_PROOF_HTML_RENDERER_VERSION in html


def test_redacted_text_never_reveals_original() -> None:
    html = render_platform_proof_report(build_redacted_payload_evidence())
    assert "[REDACTED]" in html
    assert "super-secret-token-value" not in html


def test_html_injection_payload_is_escaped() -> None:
    html = render_platform_proof_report(build_injection_evidence())
    assert _INJECTION not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_no_renderer_generated_external_dependencies() -> None:
    html = _render_pass()
    assert_no_external_report_dependencies(html)
    assert '<link rel="stylesheet"' not in html.lower()
    assert "<script" not in html.lower()


def test_deterministic_output() -> None:
    evidence = build_pass_evidence()
    first = render_platform_proof_report(evidence)
    second = render_platform_proof_report(evidence)
    assert first == second


def test_zero_and_multiple_scenarios() -> None:
    blocked = render_platform_proof_report(build_blocked_evidence())
    assert "No scenarios executed" in blocked
    multi = render_platform_proof_report(build_multi_scenario_evidence())
    assert "scenario-a" in multi
    assert "scenario-b" in multi


def test_report_safe_payload_fields_render() -> None:
    html = _render_pass()
    assert "SQL query arguments" in html
    assert "SELECT region" in html


def test_empty_optional_sections_do_not_crash() -> None:
    html = render_platform_proof_report(build_blocked_evidence())
    assert "No environment or dataset context recorded" in html
    assert "No final output recorded" in html


def test_renderer_does_not_embed_raw_evidence_json() -> None:
    html = _render_pass()
    assert not _RAW_EVIDENCE_DUMP_RE.search(html)
    assert "application/json" not in html


def test_no_chain_of_thought_fields() -> None:
    html = _render_pass()
    assert not _CHAIN_OF_THOUGHT_RE.search(html)


def test_ground_truth_separation_renders() -> None:
    html = _render_pass()
    assert "Ground truth known to proof" in html
    assert "Information available to model" in html


def test_write_platform_proof_report(tmp_path: Path) -> None:
    path = tmp_path / "report.html"
    written = write_platform_proof_report(build_pass_evidence(), output_path=path)
    assert written == path
    assert path.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")


def test_sumary_only_field_raises() -> None:
    from scripts.proof.intergrax_platform_proof_evidence import (
        ProofExecutionStep,
        ProofStepExecutionStatus,
        ReportSafePayload,
        ReportSafeText,
        ReportSafeVisibility,
    )

    evidence = build_pass_evidence()
    bad_step = ProofExecutionStep(
        step_index=99,
        step_id="bad",
        purpose=ReportSafeText(
            text="hidden",
            visibility=ReportSafeVisibility.SUMMARY_ONLY,
        ),
        action=proof_authored_report_safe_text("act"),
        observation=ReportSafePayload(
            summary=proof_authored_report_safe_text("ok"),
        ),
        status=ProofStepExecutionStatus.OK,
    )
    scenario = evidence.scenarios[0].model_copy(update={"steps": (bad_step,)})
    bad_evidence = evidence.model_copy(
        update={
            "scenarios": (scenario,),
            "evidence_graph": evidence.evidence_graph.model_copy(update={"nodes": (), "edges": ()}),
        }
    )
    with pytest.raises(PlatformProofReportRenderError):
        render_platform_proof_report(bad_evidence)


def test_preview_artifact_generation() -> None:
    """Optional local preview — not committed."""
    preview_dir = Path(".artifacts/proof/report-preview")
    preview_path = preview_dir / "report.html"
    write_platform_proof_report(build_pass_evidence(), output_path=preview_path)
    assert preview_path.is_file()
