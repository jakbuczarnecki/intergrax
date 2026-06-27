# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.evidence.eval_evidence_contracts import (
    EvalEvidenceArtifactKind,
    EvalEvidenceBasis,
    EvalEvidenceCheckKind,
    EvalEvidenceCheckResult,
    EvalEvidenceStatus,
    create_eval_evidence_check_result,
    validate_eval_evidence_report,
)
from intergrax.runtime.evidence.eval_evidence_runner import (
    build_eval_evidence_report,
    generate_eval_evidence_run_id,
    run_eval_evidence_checks,
    run_eval_scenario_library_evidence_check,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNNER_PATH = (
    _REPO_ROOT / "intergrax" / "runtime" / "evidence" / "eval_evidence_runner.py"
)

_REQUIRED_EVIDENCE_BASIS = [
    EvalEvidenceBasis.EXISTING_EVAL_CHECK,
    EvalEvidenceBasis.DETERMINISTIC_LOCAL,
    EvalEvidenceBasis.NO_NETWORK,
    EvalEvidenceBasis.NO_PROVIDER_CALLS,
    EvalEvidenceBasis.NO_REAL_LLM,
]


def _passed_scenario_library_result() -> EvalEvidenceCheckResult:
    return create_eval_evidence_check_result(
        check_id="scenario_library",
        check_kind=EvalEvidenceCheckKind.SCENARIO_LIBRARY,
        status=EvalEvidenceStatus.PASSED,
        title="Eval scenario library check available",
    )


def test_generate_eval_evidence_run_id_is_deterministic() -> None:
    first = generate_eval_evidence_run_id(root_label="local")
    second = generate_eval_evidence_run_id(root_label="local")
    assert first == second == "eval-evidence-run:local"


def test_build_eval_evidence_report_derives_passed_for_passed_result() -> None:
    report = build_eval_evidence_report(
        results=[_passed_scenario_library_result()],
        root_label="local",
    )
    assert report.status is EvalEvidenceStatus.PASSED
    assert report.summary == "All eval evidence checks passed."


def test_build_eval_evidence_report_derives_failed_when_one_result_failed() -> None:
    results = [
        _passed_scenario_library_result(),
        create_eval_evidence_check_result(
            check_id="regression_surface",
            check_kind=EvalEvidenceCheckKind.REGRESSION_SURFACE,
            status=EvalEvidenceStatus.FAILED,
            title="Regression surface failed",
        ),
    ]
    report = build_eval_evidence_report(results=results, root_label="local")
    assert report.status is EvalEvidenceStatus.FAILED
    assert report.summary == "One or more eval evidence checks failed."


def test_run_eval_scenario_library_evidence_check_passes_when_source_exists() -> None:
    result = run_eval_scenario_library_evidence_check(root=_REPO_ROOT, root_label="local")
    assert result.status is EvalEvidenceStatus.PASSED


def test_run_eval_scenario_library_evidence_check_includes_required_fields() -> None:
    result = run_eval_scenario_library_evidence_check(root=_REPO_ROOT, root_label="local")
    assert result.check_id == "scenario_library"
    assert result.check_kind is EvalEvidenceCheckKind.SCENARIO_LIBRARY
    assert result.basis == _REQUIRED_EVIDENCE_BASIS
    assert len(result.artifact_refs) == 1
    assert result.artifact_refs[0].kind is EvalEvidenceArtifactKind.SOURCE_CHECK
    assert result.artifact_refs[0].path == "scripts/maintenance/check_eval_scenario_library.py"
    assert result.metadata["network"] == "disabled"
    assert result.metadata["provider_calls"] == "disabled"
    assert result.metadata["real_llm_evaluation"] == "disabled"
    assert result.metadata["llm"] == "none"


def test_run_eval_scenario_library_evidence_check_unavailable_when_source_missing(
    tmp_path: Path,
) -> None:
    result = run_eval_scenario_library_evidence_check(root=tmp_path, root_label="local")
    assert result.status is EvalEvidenceStatus.UNAVAILABLE
    assert result.check_id == "scenario_library"
    assert result.check_kind is EvalEvidenceCheckKind.SCENARIO_LIBRARY
    assert result.metadata["network"] == "disabled"
    assert result.metadata["provider_calls"] == "disabled"
    assert result.metadata["real_llm_evaluation"] == "disabled"


def test_run_eval_evidence_checks_returns_valid_report() -> None:
    report = run_eval_evidence_checks(root=_REPO_ROOT, root_label="local")
    validate_eval_evidence_report(report)


def test_run_eval_evidence_checks_does_not_write_files(tmp_path: Path) -> None:
    before = set(tmp_path.rglob("*"))
    run_eval_evidence_checks(root=tmp_path, root_label="local")
    after = set(tmp_path.rglob("*"))
    assert before == after


def test_eval_evidence_runner_has_no_forbidden_imports() -> None:
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
            f"eval_evidence_runner.py contains forbidden import token: {token}"
        )


def test_eval_evidence_runner_has_no_obvious_network_imports() -> None:
    forbidden = ("requests", "httpx", "urllib", "socket")
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    for token in forbidden:
        assert token not in source, (
            f"eval_evidence_runner.py contains forbidden network import token: {token}"
        )


def test_eval_evidence_runner_does_not_execute_scenario_library_script() -> None:
    source = _RUNNER_PATH.read_text(encoding="utf-8")
    forbidden = (
        "subprocess",
        "runpy",
        "importlib",
        "check_eval_scenario_library.main",
        "exec(",
        "eval(",
    )
    for token in forbidden:
        assert token not in source, (
            f"eval_evidence_runner.py appears to execute scenario library script: {token}"
        )
