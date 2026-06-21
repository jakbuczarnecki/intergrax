# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.cost_evidence_contracts import (
    CostEvidenceArtifactKind,
    CostEvidenceBasis,
    CostEvidenceCheckKind,
    CostEvidenceCheckResult,
    CostEvidenceStatus,
    create_cost_evidence_check_result,
    validate_cost_evidence_report,
)
from intergrax.runtime.evidence.cost_evidence_runner import (
    build_cost_evidence_report,
    generate_cost_evidence_run_id,
    run_cost_evidence_checks,
    run_trace_budget_facets_cost_check,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNNER_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "evidence" / "cost_evidence_runner.py"
)

_REQUIRED_EVIDENCE_BASIS = [
    CostEvidenceBasis.LOCAL_TRACE_BUDGET,
    CostEvidenceBasis.DETERMINISTIC_LOCAL,
    CostEvidenceBasis.NO_PROVIDER_PRICING,
    CostEvidenceBasis.NO_REAL_LLM_METERING,
    CostEvidenceBasis.NO_NETWORK,
]


def _passed_trace_budget_facets_result() -> CostEvidenceCheckResult:
    return create_cost_evidence_check_result(
        check_id="trace_budget_facets",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Trace budget facets available",
    )


def test_generate_cost_evidence_run_id_is_deterministic() -> None:
    first = generate_cost_evidence_run_id(root_label="local")
    second = generate_cost_evidence_run_id(root_label="local")
    assert first == second == "cost-evidence-run:local"


def test_build_cost_evidence_report_derives_passed_for_passed_result() -> None:
    report = build_cost_evidence_report(
        results=[_passed_trace_budget_facets_result()],
        root_label="local",
    )
    assert report.status is CostEvidenceStatus.PASSED
    assert report.summary == "All cost evidence checks passed."


def test_build_cost_evidence_report_derives_failed_when_one_result_failed() -> None:
    results = [
        _passed_trace_budget_facets_result(),
        create_cost_evidence_check_result(
            check_id="core_budget_signals",
            check_kind=CostEvidenceCheckKind.CORE_BUDGET_SIGNALS,
            status=CostEvidenceStatus.FAILED,
            title="Core budget signals failed",
        ),
    ]
    report = build_cost_evidence_report(results=results, root_label="local")
    assert report.status is CostEvidenceStatus.FAILED
    assert report.summary == "One or more cost evidence checks failed."


def test_run_trace_budget_facets_cost_check_returns_passed() -> None:
    result = run_trace_budget_facets_cost_check(root_label="local")
    assert result.status is CostEvidenceStatus.PASSED


def test_run_trace_budget_facets_cost_check_includes_required_fields() -> None:
    result = run_trace_budget_facets_cost_check(root_label="local")
    assert result.check_id == "trace_budget_facets"
    assert result.check_kind is CostEvidenceCheckKind.TRACE_BUDGET_FACETS
    assert result.basis == _REQUIRED_EVIDENCE_BASIS
    assert len(result.artifact_refs) == 1
    assert result.artifact_refs[0].kind is CostEvidenceArtifactKind.SOURCE_CHECK
    assert (
        result.artifact_refs[0].path
        == "intergrax/runtime/evidence/trace_timeline_facets.py"
    )
    assert result.metadata["provider_pricing"] == "disabled"
    assert result.metadata["billing"] == "disabled"
    assert result.metadata["cloud_cost_estimation"] == "disabled"
    assert result.metadata["real_llm_metering"] == "disabled"
    assert result.metadata["network"] == "disabled"


def test_run_cost_evidence_checks_returns_valid_report() -> None:
    report = run_cost_evidence_checks(root_label="local")
    validate_cost_evidence_report(report)


def test_run_cost_evidence_checks_does_not_write_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    before = set(tmp_path.rglob("*"))
    run_cost_evidence_checks(root_label="local")
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_cost_evidence_runner_has_no_forbidden_imports() -> None:
    forbidden = (
        "applications.",
        "agents.",
        "from applications",
        "from agents",
        "intergrax.cli",
        "from intergrax.cli",
    )
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_runner.py contains forbidden import token: {token}"
        )


def test_cost_evidence_runner_has_no_obvious_network_imports() -> None:
    forbidden = ("requests", "httpx", "urllib", "socket")
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_runner.py contains forbidden network import token: {token}"
        )


def test_cost_evidence_runner_has_no_billing_or_provider_pricing_tokens() -> None:
    forbidden = (
        "price_per_token",
        "stripe",
        "invoice",
        "openai",
        "anthropic",
    )
    source = _RUNNER_PATH.read_text(encoding="utf-8").lower()
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_runner.py contains forbidden token: {token}"
        )


def test_cost_evidence_runner_does_not_execute_trace_export_or_cli() -> None:
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "subprocess",
        "runpy",
        "intergrax trace",
        "trace export",
    )
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_runner.py appears to execute trace export or CLI: {token}"
        )
