# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.cost_evidence_contracts import (
    COST_EVIDENCE_KIND,
    COST_EVIDENCE_OUTPUT_DIR,
    COST_EVIDENCE_REPORT_JSON,
    COST_EVIDENCE_REPORT_MARKDOWN,
    COST_EVIDENCE_SCHEMA_VERSION,
    CostEvidenceArtifactKind,
    CostEvidenceArtifactRef,
    CostEvidenceBasis,
    CostEvidenceCheckKind,
    CostEvidenceCheckResult,
    CostEvidenceReport,
    CostEvidenceStatus,
    create_cost_evidence_check_result,
    derive_cost_evidence_report_status,
    generate_cost_evidence_report_id,
    validate_cost_evidence_report,
)

pytestmark = pytest.mark.unit

_CONTRACTS_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "evidence"
    / "cost_evidence_contracts.py"
)


def _passed_trace_budget_facets_result() -> CostEvidenceCheckResult:
    return create_cost_evidence_check_result(
        check_id="trace_budget_facets",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Trace budget facets check passed",
    )


def _valid_report(
    *,
    results: list | None = None,
    status: CostEvidenceStatus | None = None,
) -> CostEvidenceReport:
    check_results = results or [_passed_trace_budget_facets_result()]
    report_status = status or derive_cost_evidence_report_status(check_results)
    return CostEvidenceReport(
        report_id=generate_cost_evidence_report_id(root_label="local"),
        status=report_status,
        results=check_results,
    )


def test_generate_cost_evidence_report_id_is_deterministic() -> None:
    first = generate_cost_evidence_report_id(root_label="local")
    second = generate_cost_evidence_report_id(root_label="local")
    assert first == second == "cost-evidence:local"


def test_constants_match_planned_output_path_and_filenames() -> None:
    assert COST_EVIDENCE_SCHEMA_VERSION == "1.0.0"
    assert COST_EVIDENCE_KIND == "cost_evidence"
    assert COST_EVIDENCE_OUTPUT_DIR == "build/evidence/cost"
    assert COST_EVIDENCE_REPORT_JSON == "report.json"
    assert COST_EVIDENCE_REPORT_MARKDOWN == "report.md"


def test_create_cost_evidence_check_result_uses_default_basis_and_empty_collections() -> None:
    result = create_cost_evidence_check_result(
        check_id="trace_budget_facets",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Trace budget facets check",
    )
    assert result.basis == [
        CostEvidenceBasis.LOCAL_TRACE_BUDGET,
        CostEvidenceBasis.DETERMINISTIC_LOCAL,
        CostEvidenceBasis.NO_PROVIDER_PRICING,
        CostEvidenceBasis.NO_REAL_LLM_METERING,
        CostEvidenceBasis.NO_NETWORK,
    ]
    assert result.artifact_refs == []
    assert result.metadata == {}


def test_derive_cost_evidence_report_status_empty() -> None:
    assert derive_cost_evidence_report_status([]) is CostEvidenceStatus.UNAVAILABLE


def test_derive_cost_evidence_report_status_all_passed() -> None:
    results = [_passed_trace_budget_facets_result()]
    assert derive_cost_evidence_report_status(results) is CostEvidenceStatus.PASSED


def test_derive_cost_evidence_report_status_any_failed() -> None:
    results = [
        create_cost_evidence_check_result(
            check_id="trace_budget_facets",
            check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
            status=CostEvidenceStatus.FAILED,
            title="Trace budget facets check failed",
        )
    ]
    assert derive_cost_evidence_report_status(results) is CostEvidenceStatus.FAILED


def test_derive_cost_evidence_report_status_passed_and_skipped() -> None:
    results = [
        _passed_trace_budget_facets_result(),
        create_cost_evidence_check_result(
            check_id="core_budget_signals",
            check_kind=CostEvidenceCheckKind.CORE_BUDGET_SIGNALS,
            status=CostEvidenceStatus.SKIPPED,
            title="Core budget signals skipped",
        ),
    ]
    assert derive_cost_evidence_report_status(results) is CostEvidenceStatus.SKIPPED


def test_validate_cost_evidence_report_valid_trace_budget_facets_passed() -> None:
    validate_cost_evidence_report(_valid_report())


def test_validate_cost_evidence_report_rejects_empty_report_id() -> None:
    report = _valid_report()
    report = report.model_copy(update={"report_id": "   "})
    with pytest.raises(ValueError, match="report_id must not be empty"):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_empty_results() -> None:
    report = CostEvidenceReport(
        report_id=generate_cost_evidence_report_id(root_label="local"),
        status=CostEvidenceStatus.UNAVAILABLE,
        results=[],
    )
    with pytest.raises(ValueError, match="results must not be empty"):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_empty_check_id() -> None:
    result = create_cost_evidence_check_result(
        check_id="   ",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Missing check id",
    )
    report = _valid_report(results=[result])
    with pytest.raises(ValueError, match="check_id must not be empty"):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_duplicate_check_id() -> None:
    duplicate = _passed_trace_budget_facets_result()
    report = _valid_report(results=[duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate check_id: trace_budget_facets"):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_missing_required_safety_basis() -> None:
    result = create_cost_evidence_check_result(
        check_id="trace_budget_facets",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Missing safety basis",
        basis=[CostEvidenceBasis.LOCAL_TRACE_BUDGET],
    )
    report = _valid_report(results=[result])
    with pytest.raises(ValueError, match="result must include DETERMINISTIC_LOCAL"):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_missing_local_source_basis() -> None:
    result = create_cost_evidence_check_result(
        check_id="trace_budget_facets",
        check_kind=CostEvidenceCheckKind.TRACE_BUDGET_FACETS,
        status=CostEvidenceStatus.PASSED,
        title="Missing local source basis",
        basis=[
            CostEvidenceBasis.DETERMINISTIC_LOCAL,
            CostEvidenceBasis.NO_PROVIDER_PRICING,
            CostEvidenceBasis.NO_REAL_LLM_METERING,
            CostEvidenceBasis.NO_NETWORK,
        ],
    )
    report = _valid_report(results=[result])
    with pytest.raises(
        ValueError,
        match="result must include at least one local source basis",
    ):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_empty_artifact_path() -> None:
    result = _passed_trace_budget_facets_result()
    result = result.model_copy(
        update={
            "artifact_refs": [
                CostEvidenceArtifactRef(
                    kind=CostEvidenceArtifactKind.SOURCE_CHECK,
                    path="   ",
                )
            ]
        }
    )
    report = _valid_report(results=[result])
    with pytest.raises(
        ValueError,
        match="result artifact path must not be empty \\(trace_budget_facets\\)",
    ):
        validate_cost_evidence_report(report)


def test_validate_cost_evidence_report_rejects_status_mismatch() -> None:
    report = _valid_report(status=CostEvidenceStatus.FAILED)
    with pytest.raises(ValueError, match="report status must be PASSED"):
        validate_cost_evidence_report(report)


def test_cost_evidence_contracts_have_no_forbidden_imports() -> None:
    forbidden = (
        "applications.",
        "agents.",
        "from applications",
        "from agents",
        "intergrax.cli",
        "from intergrax.cli",
    )
    source = _CONTRACTS_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_contracts.py contains forbidden import token: {token}"
        )


def test_cost_evidence_contracts_have_no_network_imports() -> None:
    forbidden = ("requests", "httpx", "urllib", "socket")
    source = _CONTRACTS_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_contracts.py contains forbidden network import token: {token}"
        )


def test_cost_evidence_contracts_have_no_billing_or_provider_pricing_tokens() -> None:
    forbidden = (
        "price_per_token",
        "billing",
        "invoice",
        "stripe",
        "openai",
        "anthropic",
    )
    source = _CONTRACTS_PATH.read_text(encoding="utf-8").lower()
    for token in forbidden:
        assert token not in source, (
            f"cost_evidence_contracts.py contains forbidden token: {token}"
        )
