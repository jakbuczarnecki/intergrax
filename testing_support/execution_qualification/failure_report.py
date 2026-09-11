# © Artur Czarnecki. All rights reserved.

"""Deterministic failure projection for execution qualification runs."""

from __future__ import annotations

from collections.abc import Mapping

from testing_support.execution_qualification.contracts import (
    ExecutionQualificationRunResult,
    ExecutionQualificationSuiteResult,
    QualificationRunStatus,
    QualificationSuiteStatus,
)


def format_execution_qualification_failure(
    result: ExecutionQualificationRunResult,
    *,
    label_by_suite_id: Mapping[str, str],
) -> str:
    """Build a maintainer-facing summary; log paths are canonical detailed evidence."""
    lines = ["Execution qualification failed:"]
    for suite_result in result.suite_results:
        label = label_by_suite_id.get(suite_result.suite_id, suite_result.suite_id)
        lines.append("")
        lines.append(f"[{label}]")
        lines.append(f"status={suite_result.status.value}")
        lines.append(f"outcome={suite_result.outcome_kind.value}")
        exit_code = suite_result.exit_code
        lines.append(f"exit_code={exit_code if exit_code is not None else 'null'}")
        lines.append(f"log={suite_result.log_path}")
    return "\n".join(lines)


def assert_execution_qualification_pass(
    result: ExecutionQualificationRunResult,
    *,
    label_by_suite_id: Mapping[str, str],
) -> None:
    if result.status is QualificationRunStatus.PASS:
        return
    failing = [
        suite_result
        for suite_result in result.suite_results
        if suite_result.status is not QualificationSuiteStatus.PASS
    ]
    if not failing:
        raise AssertionError(
            "aggregate qualification status is FAIL but no non-PASS suite results were recorded"
        )
    raise AssertionError(
        format_execution_qualification_failure(result, label_by_suite_id=label_by_suite_id)
    )
