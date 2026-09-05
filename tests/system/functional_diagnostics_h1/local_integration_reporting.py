# © Artur Czarnecki. All rights reserved.

"""Machine and human report serialization for DIAG-H1-K qualification."""

from __future__ import annotations

import json
from pathlib import Path

from tests.system.functional_diagnostics_h1.models import (
    HealthVerdict,
    H1_K_SCHEMA_VERSION,
    LocalIntegrationQualificationReport,
    LocalIntegrationRunResult,
    LocalIntegrationSuiteResult,
    PytestFailureEvidence,
)
from tests.system.functional_diagnostics_h1.qualification_spec import (
    resolve_local_integration_qualification_spec,
)


def _failure_evidence_to_json(
    evidence: PytestFailureEvidence,
) -> dict[str, str | None]:
    return {
        "stdout_tail": evidence.stdout_tail,
        "stderr_tail": evidence.stderr_tail,
        "basetemp_path": evidence.basetemp_path,
        "failure_phase": (
            evidence.failure_phase.value if evidence.failure_phase is not None else None
        ),
    }


def _suite_to_json(
    suite: LocalIntegrationSuiteResult,
) -> dict[str, str | int | float | None | dict[str, str | None]]:
    payload: dict[str, str | int | float | None | dict[str, str | None]] = {
        "target": suite.target,
        "collected": suite.collected,
        "passed": suite.passed,
        "failed": suite.failed,
        "skipped": suite.skipped,
        "xfailed": suite.xfailed,
        "xpassed": suite.xpassed,
        "errors": suite.errors,
        "collection_errors": suite.collection_errors,
        "exit_code": suite.exit_code,
        "duration_seconds": suite.duration_seconds,
        "verdict": suite.verdict.value,
        "dependency_class": suite.dependency_class.value,
    }
    if suite.failure_evidence is not None:
        payload["failure_evidence"] = _failure_evidence_to_json(suite.failure_evidence)
    return payload


def _run_to_json(run: LocalIntegrationRunResult) -> dict[str, str | int | list[dict[str, str | int | float | None]] | None]:
    return {
        "run_index": run.run_index,
        "suite_results": [_suite_to_json(item) for item in run.suite_results],
        "verdict": run.verdict.value,
        "total_collected": run.total_collected,
        "total_passed": run.total_passed,
        "total_failed": run.total_failed,
        "total_errors": run.total_errors,
    }


def local_integration_qualification_report_to_json(
    report: LocalIntegrationQualificationReport,
) -> dict[str, str | bool | list[dict[str, str | int | list[dict[str, str | int | float | None]] | None]] | tuple[str, ...]]:
    return {
        "qualification_id": report.qualification_id,
        "schema_version": report.schema_version,
        "tested_sha": report.tested_sha,
        "start_head": report.start_head,
        "final_head": report.final_head,
        "origin_development_at_start": report.origin_development_at_start,
        "origin_development_at_end": report.origin_development_at_end,
        "working_tree_clean_at_start": report.working_tree_clean_at_start,
        "working_tree_clean_at_end": report.working_tree_clean_at_end,
        "repository_precondition": report.repository_precondition.value,
        "repository_postcondition": report.repository_postcondition.value,
        "runs": [_run_to_json(item) for item in report.runs],
        "repeatability_verdict": report.repeatability_verdict.value,
        "overall_verdict": report.overall_verdict.value,
        "blocking_findings": list(report.blocking_findings),
        "timestamp": report.timestamp,
    }


def validate_local_integration_report_integrity(
    report: LocalIntegrationQualificationReport,
    *,
    artifact_directory: Path | None = None,
) -> tuple[HealthVerdict, tuple[str, ...]]:
    violations: list[str] = []
    if artifact_directory is not None:
        try:
            expected_spec = resolve_local_integration_qualification_spec(
                report.qualification_id
            )
        except ValueError:
            pass
        else:
            if expected_spec.artifact_directory != artifact_directory:
                violations.append(
                    "artifact_directory_mismatch:"
                    f"{report.qualification_id}:"
                    f"expected={expected_spec.artifact_directory.as_posix()}:"
                    f"actual={artifact_directory.as_posix()}"
                )
    if report.overall_verdict is HealthVerdict.PASS:
        if report.repository_precondition is not HealthVerdict.PASS:
            violations.append("precondition_not_pass_with_overall_pass")
        if report.repository_postcondition is not HealthVerdict.PASS:
            violations.append("postcondition_not_pass_with_overall_pass")
        if report.repeatability_verdict is not HealthVerdict.PASS:
            violations.append("repeatability_not_pass_with_overall_pass")
        if report.blocking_findings:
            violations.append("blocking_findings_coexist_with_overall_pass")
        if report.start_head != report.final_head:
            violations.append(
                f"head_changed:{report.start_head}->{report.final_head}"
            )
        if not report.working_tree_clean_at_end:
            violations.append("working_tree_not_clean_at_end")
        for run in report.runs:
            if run.verdict is not HealthVerdict.PASS:
                violations.append(f"failed_run_coexist_with_overall_pass:run_{run.run_index}")
            for suite in run.suite_results:
                if suite.verdict is not HealthVerdict.PASS:
                    violations.append(
                        f"failed_suite_coexist_with_overall_pass:{suite.target}"
                    )
    verdict = HealthVerdict.PASS if not violations else HealthVerdict.FAILED
    return verdict, tuple(violations)


def build_local_integration_human_report(
    report: LocalIntegrationQualificationReport,
    *,
    artifact_json_path: str,
) -> str:
    lines = [
        f"# {report.qualification_id} RESULT",
        "",
        "## Verdict",
        report.overall_verdict.value,
        "",
        "## Historical R1",
        "FAILED_PRECONDITION (immutable)",
        "",
        "## Start HEAD",
        report.start_head,
        "",
        "## Qualified SHA",
        report.tested_sha,
        "",
        "## Final HEAD",
        report.final_head,
        "",
        "## HEAD == origin/development",
        "YES"
        if report.final_head == report.origin_development_at_end
        else "NO",
        "",
        "## Tree clean at start",
        "YES" if report.working_tree_clean_at_start else "NO",
        "",
        "## Tree clean at end",
        "YES" if report.working_tree_clean_at_end else "NO",
        "",
        "## Origin stable",
        "YES"
        if report.origin_development_at_start == report.origin_development_at_end
        else "NO",
        "",
        "## H1-K execution authority",
        "tests.system.functional_diagnostics_h1.local_integration.run_h1_k_local_integration",
        "",
        "## Target registry authority",
        "tests.system.functional_diagnostics_h1.inventory.LOCAL_INTEGRATION_TARGETS",
        "",
    ]
    for run in report.runs:
        lines.extend(
            [
                f"## Run {run.run_index}",
                "",
                "| Target | Collected | Passed | Failed | Errors | Skipped | Xfailed | Verdict |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for suite in run.suite_results:
            collected_display = (
                "unknown" if suite.collected is None else str(suite.collected)
            )
            lines.append(
                f"| {suite.target} | {collected_display} | {suite.passed} | "
                f"{suite.failed} | {suite.errors} | {suite.skipped} | "
                f"{suite.xfailed} | {suite.verdict.value} |"
            )
        lines.append("")
    target_set_stable = all(
        tuple(item.target for item in run.suite_results)
        == tuple(item.target for item in report.runs[0].suite_results)
        for run in report.runs
    )
    lines.extend(
        [
            "## Target-set stability",
            "PASS" if target_set_stable else "FAIL",
            "",
            "## Count stability",
            report.repeatability_verdict.value
            if report.repeatability_verdict is HealthVerdict.PASS
            else "FAIL",
            "",
            "## Outcome stability",
            report.repeatability_verdict.value
            if report.repeatability_verdict is HealthVerdict.PASS
            else "FAIL",
            "",
            "## Repeatability",
            report.repeatability_verdict.value,
            "",
            "## Blocking findings",
            "NONE" if not report.blocking_findings else "\n".join(
                f"- {item}" for item in report.blocking_findings
            ),
            "",
            "## Machine artifact",
            artifact_json_path,
            "",
            "## Human artifact",
            artifact_json_path.replace("qualification-report.json", "qualification-report.md"),
            "",
            "## Summary",
            "",
        ]
    )
    for run in report.runs:
        lines.append(f"Run {run.run_index} {run.verdict.value}")
    lines.append(f"Repeatability {report.repeatability_verdict.value}")
    lines.append(
        f"Repository integrity {report.repository_postcondition.value}"
    )
    lines.append(f"Overall {report.overall_verdict.value}")
    if report.overall_verdict is HealthVerdict.PASS:
        lines.extend(
            [
                "",
                "## Final statement",
                "",
                "H1-K LOCAL DIAGNOSTIC INTEGRATION",
                "= QUALIFIED",
                "",
                f"LOCAL SUITES",
                f"= {len(report.runs[0].suite_results)}/{len(report.runs[0].suite_results)} PASS",
                "",
                "THREE-RUN REPEATABILITY",
                "= PASS",
                "",
                "TEST COUNTS",
                "= STABLE",
                "",
                "QUALIFICATION TREE",
                "= CLEAN",
                "",
                f"QUALIFIED SHA",
                f"= {report.tested_sha}",
                "",
                "H1-R3 CLEAN CANONICAL",
                "= READY",
            ]
        )
    return "\n".join(lines) + "\n"


def write_local_integration_qualification_report(
    path: Path,
    report: LocalIntegrationQualificationReport,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = local_integration_qualification_report_to_json(report)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def write_local_integration_human_report(
    path: Path,
    report: LocalIntegrationQualificationReport,
    *,
    artifact_json_path: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_local_integration_human_report(
            report,
            artifact_json_path=artifact_json_path,
        ),
        encoding="utf-8",
    )
