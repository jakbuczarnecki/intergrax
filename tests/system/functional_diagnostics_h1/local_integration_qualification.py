# © Artur Czarnecki. All rights reserved.

"""DIAG-H1-K-QUALIFICATION-R1 canonical local integration qualification runner."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (_REPO_ROOT,):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from tests.system.functional_diagnostics_h1.local_integration import (
    evaluate_local_integration_repeatability,
    run_h1_k_local_integration,
)
from tests.system.functional_diagnostics_h1.local_integration_reporting import (
    validate_local_integration_report_integrity,
    write_local_integration_human_report,
    write_local_integration_qualification_report,
)
from tests.system.functional_diagnostics_h1.models import (
    HealthVerdict,
    H1_K_QUALIFICATION_ID,
    H1_K_SCHEMA_VERSION,
    LocalIntegrationQualificationReport,
    LocalIntegrationRunResult,
    QualificationRepositoryTransition,
)
from tests.system.functional_diagnostics_h1.qualification_spec import (
    H1_K_QUALIFICATION_SPEC,
)
from tests.system.functional_diagnostics_h1.reporting import utc_now_iso
from tests.system.functional_diagnostics_h1.repository_state import (
    assert_qualification_repository_postconditions,
    assert_qualification_repository_state,
    capture_qualification_repository_state,
)

_EXIT_PASS = 0
_EXIT_FAILED = 1
_EXIT_FAILED_PRECONDITION = 3

_CANONICAL_RUN_COUNT = 3


def _aggregate_overall_verdict(
    *,
    repository_precondition: HealthVerdict,
    repository_postcondition: HealthVerdict,
    runs: tuple[LocalIntegrationRunResult, ...],
    repeatability_verdict: HealthVerdict,
    blocking_findings: tuple[str, ...],
) -> HealthVerdict:
    if repository_precondition is HealthVerdict.FAILED_PRECONDITION:
        return HealthVerdict.FAILED_PRECONDITION
    if blocking_findings:
        return HealthVerdict.FAILED
    if repository_precondition is not HealthVerdict.PASS:
        return HealthVerdict.FAILED
    if repository_postcondition is not HealthVerdict.PASS:
        return HealthVerdict.FAILED
    if any(run.verdict is not HealthVerdict.PASS for run in runs):
        return HealthVerdict.FAILED
    if repeatability_verdict is not HealthVerdict.PASS:
        return HealthVerdict.FAILED
    return HealthVerdict.PASS


def run_h1_k_qualification(
    *,
    artifact_dir: Path | None = None,
    run_count: int = _CANONICAL_RUN_COUNT,
) -> int:
    spec = H1_K_QUALIFICATION_SPEC
    resolved_artifact_dir = artifact_dir or spec.artifact_directory
    start_state = capture_qualification_repository_state(_REPO_ROOT)
    repository_precondition, precondition_violations = assert_qualification_repository_state(
        start_state,
        requires_clean_repository=spec.requires_clean_repository,
        requires_origin_development_match=spec.requires_origin_development_match,
    )
    if repository_precondition is HealthVerdict.FAILED_PRECONDITION:
        report = LocalIntegrationQualificationReport(
            qualification_id=H1_K_QUALIFICATION_ID,
            schema_version=H1_K_SCHEMA_VERSION,
            tested_sha=start_state.head_sha,
            start_head=start_state.head_sha,
            final_head=start_state.head_sha,
            origin_development_at_start=start_state.origin_development_sha,
            origin_development_at_end=start_state.origin_development_sha,
            working_tree_clean_at_start=start_state.working_tree_clean,
            working_tree_clean_at_end=start_state.working_tree_clean,
            repository_precondition=repository_precondition,
            repository_postcondition=HealthVerdict.FAILED,
            runs=(),
            repeatability_verdict=HealthVerdict.FAILED,
            overall_verdict=HealthVerdict.FAILED_PRECONDITION,
            blocking_findings=precondition_violations,
            timestamp=utc_now_iso(),
        )
        json_path = resolved_artifact_dir / "qualification-report.json"
        write_local_integration_qualification_report(json_path, report)
        write_local_integration_human_report(
            resolved_artifact_dir / "qualification-report.md",
            report,
            artifact_json_path=json_path.as_posix(),
        )
        return _EXIT_FAILED_PRECONDITION

    runs: list[LocalIntegrationRunResult] = []
    blocking_findings: list[str] = list(precondition_violations)
    for index in range(run_count):
        runs.append(run_h1_k_local_integration(run_index=index + 1))

    repeatability_verdict, repeatability_findings = evaluate_local_integration_repeatability(
        tuple(runs)
    )
    blocking_findings.extend(repeatability_findings)

    end_state = capture_qualification_repository_state(_REPO_ROOT)
    transition = QualificationRepositoryTransition(start=start_state, end=end_state)
    repository_postcondition, postcondition_violations = (
        assert_qualification_repository_postconditions(transition, spec)
    )
    blocking_findings.extend(postcondition_violations)

    overall_verdict = _aggregate_overall_verdict(
        repository_precondition=repository_precondition,
        repository_postcondition=repository_postcondition,
        runs=tuple(runs),
        repeatability_verdict=repeatability_verdict,
        blocking_findings=tuple(blocking_findings),
    )

    report = LocalIntegrationQualificationReport(
        qualification_id=H1_K_QUALIFICATION_ID,
        schema_version=H1_K_SCHEMA_VERSION,
        tested_sha=start_state.head_sha,
        start_head=start_state.head_sha,
        final_head=end_state.head_sha,
        origin_development_at_start=start_state.origin_development_sha,
        origin_development_at_end=end_state.origin_development_sha,
        working_tree_clean_at_start=start_state.working_tree_clean,
        working_tree_clean_at_end=end_state.working_tree_clean,
        repository_precondition=repository_precondition,
        repository_postcondition=repository_postcondition,
        runs=tuple(runs),
        repeatability_verdict=repeatability_verdict,
        overall_verdict=overall_verdict,
        blocking_findings=tuple(blocking_findings),
        timestamp=utc_now_iso(),
    )

    integrity_verdict, integrity_violations = validate_local_integration_report_integrity(
        report
    )
    if integrity_verdict is not HealthVerdict.PASS:
        blocking_findings.extend(integrity_violations)
        overall_verdict = HealthVerdict.FAILED
        report = LocalIntegrationQualificationReport(
            qualification_id=report.qualification_id,
            schema_version=report.schema_version,
            tested_sha=report.tested_sha,
            start_head=report.start_head,
            final_head=report.final_head,
            origin_development_at_start=report.origin_development_at_start,
            origin_development_at_end=report.origin_development_at_end,
            working_tree_clean_at_start=report.working_tree_clean_at_start,
            working_tree_clean_at_end=report.working_tree_clean_at_end,
            repository_precondition=report.repository_precondition,
            repository_postcondition=report.repository_postcondition,
            runs=report.runs,
            repeatability_verdict=report.repeatability_verdict,
            overall_verdict=overall_verdict,
            blocking_findings=tuple(blocking_findings),
            timestamp=report.timestamp,
        )

    json_path = resolved_artifact_dir / "qualification-report.json"
    write_local_integration_qualification_report(json_path, report)
    write_local_integration_human_report(
        resolved_artifact_dir / "qualification-report.md",
        report,
        artifact_json_path=json_path.as_posix(),
    )

    if overall_verdict is HealthVerdict.PASS:
        return _EXIT_PASS
    if overall_verdict is HealthVerdict.FAILED_PRECONDITION:
        return _EXIT_FAILED_PRECONDITION
    return _EXIT_FAILED


def main() -> None:
    exit_code = run_h1_k_qualification()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
