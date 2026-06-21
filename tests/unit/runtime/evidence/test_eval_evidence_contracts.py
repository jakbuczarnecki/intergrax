# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.eval_evidence_contracts import (
    EVAL_EVIDENCE_KIND,
    EVAL_EVIDENCE_OUTPUT_DIR,
    EVAL_EVIDENCE_REPORT_JSON,
    EVAL_EVIDENCE_REPORT_MARKDOWN,
    EVAL_EVIDENCE_SCHEMA_VERSION,
    EvalEvidenceArtifactKind,
    EvalEvidenceArtifactRef,
    EvalEvidenceBasis,
    EvalEvidenceCheckKind,
    EvalEvidenceCheckResult,
    EvalEvidenceReport,
    EvalEvidenceStatus,
    create_eval_evidence_check_result,
    derive_eval_evidence_report_status,
    generate_eval_evidence_report_id,
    validate_eval_evidence_report,
)

pytestmark = pytest.mark.unit

_CONTRACTS_PATH = (
    Path(__file__).resolve().parents[4]
    / "intergrax"
    / "runtime"
    / "evidence"
    / "eval_evidence_contracts.py"
)


def _passed_scenario_library_result() -> EvalEvidenceCheckResult:
    return create_eval_evidence_check_result(
        check_id="scenario_library",
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.PASSED,
        title="Scenario library check passed",
    )


def _valid_report(
    *,
    results: list | None = None,
    status: EvalEvidenceStatus | None = None,
) -> EvalEvidenceReport:
    check_results = results or [_passed_scenario_library_result()]
    report_status = status or derive_eval_evidence_report_status(check_results)
    return EvalEvidenceReport(
        report_id=generate_eval_evidence_report_id(root_label="local"),
        status=report_status,
        results=check_results,
    )


def test_generate_eval_evidence_report_id_is_deterministic() -> None:
    first = generate_eval_evidence_report_id(root_label="local")
    second = generate_eval_evidence_report_id(root_label="local")
    assert first == second == "eval-evidence:local"


def test_constants_match_planned_output_path_and_filenames() -> None:
    assert EVAL_EVIDENCE_SCHEMA_VERSION == "1.0.0"
    assert EVAL_EVIDENCE_KIND == "eval_regression_evidence"
    assert EVAL_EVIDENCE_OUTPUT_DIR == "build/evidence/eval"
    assert EVAL_EVIDENCE_REPORT_JSON == "report.json"
    assert EVAL_EVIDENCE_REPORT_MARKDOWN == "report.md"


def test_create_eval_evidence_check_result_uses_default_basis_and_empty_collections() -> None:
    result = create_eval_evidence_check_result(
        check_id="scenario_library",
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.PASSED,
        title="Scenario library check",
    )
    assert result.basis == [
        EvalEvidenceBasis.EXISTING_EVAL_CHECK,
        EvalEvidenceBasis.DETERMINISTIC_LOCAL,
        EvalEvidenceBasis.NO_NETWORK,
        EvalEvidenceBasis.NO_PROVIDER_CALLS,
        EvalEvidenceBasis.NO_REAL_LLM,
    ]
    assert result.artifact_refs == []
    assert result.metadata == {}


def test_derive_eval_evidence_report_status_empty() -> None:
    assert derive_eval_evidence_report_status([]) is EvalEvidenceStatus.UNAVAILABLE


def test_derive_eval_evidence_report_status_all_passed() -> None:
    results = [_passed_scenario_library_result()]
    assert derive_eval_evidence_report_status(results) is EvalEvidenceStatus.PASSED


def test_derive_eval_evidence_report_status_any_failed() -> None:
    results = [
        create_eval_evidence_check_result(
            check_id="scenario_library",
            check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
            status=EvalEvidenceStatus.FAILED,
            title="Scenario library check failed",
        )
    ]
    assert derive_eval_evidence_report_status(results) is EvalEvidenceStatus.FAILED


def test_derive_eval_evidence_report_status_passed_and_skipped() -> None:
    results = [
        _passed_scenario_library_result(),
        create_eval_evidence_check_result(
            check_id="regression_surface",
            check_kind=EvalEvidenceCheckKind.REGRESSION_SURFACE,
            status=EvalEvidenceStatus.SKIPPED,
            title="Regression surface skipped",
        ),
    ]
    assert derive_eval_evidence_report_status(results) is EvalEvidenceStatus.SKIPPED


def test_validate_eval_evidence_report_valid_scenario_library_passed() -> None:
    validate_eval_evidence_report(_valid_report())


def test_validate_eval_evidence_report_rejects_empty_report_id() -> None:
    report = _valid_report()
    report = report.model_copy(update={"report_id": "   "})
    with pytest.raises(ValueError, match="report_id must not be empty"):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_empty_results() -> None:
    report = EvalEvidenceReport(
        report_id=generate_eval_evidence_report_id(root_label="local"),
        status=EvalEvidenceStatus.UNAVAILABLE,
        results=[],
    )
    with pytest.raises(ValueError, match="results must not be empty"):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_empty_check_id() -> None:
    result = create_eval_evidence_check_result(
        check_id="   ",
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.PASSED,
        title="Missing check id",
    )
    report = _valid_report(results=[result])
    with pytest.raises(ValueError, match="check_id must not be empty"):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_duplicate_check_id() -> None:
    duplicate = _passed_scenario_library_result()
    report = _valid_report(results=[duplicate, duplicate])
    with pytest.raises(ValueError, match="duplicate check_id: scenario_library"):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_missing_required_basis() -> None:
    result = create_eval_evidence_check_result(
        check_id="scenario_library",
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.PASSED,
        title="Missing basis",
        basis=[EvalEvidenceBasis.EXISTING_EVAL_CHECK],
    )
    report = _valid_report(results=[result])
    with pytest.raises(ValueError, match="result must include DETERMINISTIC_LOCAL"):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_empty_artifact_path() -> None:
    result = _passed_scenario_library_result()
    result = result.model_copy(
        update={
            "artifact_refs": [
                EvalEvidenceArtifactRef(
                    kind=EvalEvidenceArtifactKind.SOURCE_CHECK,
                    path="   ",
                )
            ]
        }
    )
    report = _valid_report(results=[result])
    with pytest.raises(
        ValueError,
        match="result artifact path must not be empty \\(scenario_library\\)",
    ):
        validate_eval_evidence_report(report)


def test_validate_eval_evidence_report_rejects_status_mismatch() -> None:
    report = _valid_report(status=EvalEvidenceStatus.FAILED)
    with pytest.raises(ValueError, match="report status must be PASSED"):
        validate_eval_evidence_report(report)


def test_eval_evidence_contracts_have_no_forbidden_imports() -> None:
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
            f"eval_evidence_contracts.py contains forbidden import token: {token}"
        )


def test_eval_evidence_contracts_have_no_network_imports() -> None:
    forbidden = ("requests", "httpx", "urllib", "socket")
    source = _CONTRACTS_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"eval_evidence_contracts.py contains forbidden network import token: {token}"
        )
