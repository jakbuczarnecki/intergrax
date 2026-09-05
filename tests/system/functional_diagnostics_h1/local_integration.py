# © Artur Czarnecki. All rights reserved.

"""Reusable H1-K local diagnostic integration execution core."""

from __future__ import annotations

from tests.system.functional_diagnostics_h1.inventory import LOCAL_INTEGRATION_TARGETS
from tests.system.functional_diagnostics_h1.models import (
    GateResult,
    HealthGateId,
    HealthVerdict,
    LocalIntegrationDependencyClass,
    LocalIntegrationRunResult,
    LocalIntegrationSuiteResult,
    PytestFailureEvidence,
    PytestSubprocessResult,
)
from tests.system.functional_diagnostics_h1.subprocess_pytest import (
    classify_pytest_exit,
    run_pytest_subprocess,
)

LOCAL_INTEGRATION_ENV_OVERRIDES: dict[str, str] = {
    "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET": (
        "h1-local-integration-diagnostic-cursor-secret"
    ),
}

_LOCAL_INTEGRATION_TIMEOUT_SECONDS = 900.0


def validate_local_integration_metrics(
    result: PytestSubprocessResult,
) -> tuple[bool, str | None]:
    executed = result.passed + result.failed + result.skipped + result.xfailed + result.xpassed
    if result.passed > 0 and result.collected_count == 0:
        return False, "passed_gt_zero_but_collected_zero"
    if result.passed > 0 and result.collected_count is None:
        return False, "passed_gt_zero_but_collected_unknown"
    if result.collected_count is not None and executed > 0 and result.collected_count < executed:
        return False, "collected_lt_executed"
    if executed > 0 and (result.collected_count is None or result.collected_count == 0):
        return False, "executed_but_collected_unknown_or_zero"
    return True, None


def evaluate_local_integration_suite(
    target: str,
    result: PytestSubprocessResult,
) -> LocalIntegrationSuiteResult:
    exit_outcome = classify_pytest_exit(result)
    metrics_ok, metric_note = validate_local_integration_metrics(result)
    violations: list[str] = []
    if exit_outcome != "PASS":
        violations.append(f"exit_classification:{exit_outcome}")
    if result.failed > 0:
        violations.append("failed_gt_zero")
    if result.errors > 0:
        violations.append("errors_gt_zero")
    if result.collection_errors > 0:
        violations.append("collection_errors_gt_zero")
    if not metrics_ok and metric_note is not None:
        violations.append(metric_note)
    if result.skipped > 0:
        violations.append("unexpected_skipped")
    if result.xfailed > 0:
        violations.append("unexpected_xfailed")
    if result.xpassed > 0:
        violations.append("unexpected_xpassed")
    verdict = HealthVerdict.PASS if not violations else HealthVerdict.FAILED
    failure_evidence: PytestFailureEvidence | None = None
    if verdict is not HealthVerdict.PASS:
        failure_evidence = PytestFailureEvidence(
            stdout_tail=result.stdout_tail,
            stderr_tail=result.stderr_tail,
            basetemp_path=result.basetemp_path or "",
            failure_phase=result.failure_phase,
        )
    return LocalIntegrationSuiteResult(
        target=target,
        collected=result.collected_count,
        passed=result.passed,
        failed=result.failed,
        skipped=result.skipped,
        xfailed=result.xfailed,
        xpassed=result.xpassed,
        errors=result.errors,
        collection_errors=result.collection_errors,
        exit_code=result.exit_code,
        duration_seconds=result.duration_seconds,
        verdict=verdict,
        dependency_class=LocalIntegrationDependencyClass.LOCAL_DETERMINISTIC,
        failure_evidence=failure_evidence,
    )


def run_h1_k_local_integration(
    *,
    run_index: int = 1,
    targets: tuple[str, ...] | None = None,
) -> LocalIntegrationRunResult:
    resolved_targets = targets if targets is not None else LOCAL_INTEGRATION_TARGETS
    suite_results: list[LocalIntegrationSuiteResult] = []
    for target in resolved_targets:
        pytest_result = run_pytest_subprocess(
            (target,),
            timeout_seconds=_LOCAL_INTEGRATION_TIMEOUT_SECONDS,
            env_overrides=LOCAL_INTEGRATION_ENV_OVERRIDES,
        )
        suite_results.append(evaluate_local_integration_suite(target, pytest_result))
    collected_values = [
        item.collected for item in suite_results if item.collected is not None
    ]
    total_collected: int | None = sum(collected_values) if collected_values else None
    total_passed = sum(item.passed for item in suite_results)
    total_failed = sum(item.failed for item in suite_results)
    total_errors = sum(item.errors for item in suite_results)
    run_verdict = HealthVerdict.PASS
    if any(item.verdict is not HealthVerdict.PASS for item in suite_results):
        run_verdict = HealthVerdict.FAILED
    return LocalIntegrationRunResult(
        run_index=run_index,
        suite_results=tuple(suite_results),
        verdict=run_verdict,
        total_collected=total_collected,
        total_passed=total_passed,
        total_failed=total_failed,
        total_errors=total_errors,
    )


def gate_result_from_local_integration_run(
    run: LocalIntegrationRunResult,
) -> GateResult:
    details = tuple(
        f"{suite.target}:outcome={suite.verdict.value}:passed={suite.passed}:"
        f"failed={suite.failed}:errors={suite.errors}:"
        f"dependency_class={suite.dependency_class.value}"
        for suite in run.suite_results
    )
    return GateResult(
        gate_id=HealthGateId.H1_K_LOCAL_INTEGRATION,
        verdict=run.verdict,
        summary=f"local diagnostic integration (H1-K): {run.verdict.value}",
        details=details,
    )


def evaluate_local_integration_repeatability(
    runs: tuple[LocalIntegrationRunResult, ...],
) -> tuple[HealthVerdict, tuple[str, ...]]:
    if not runs:
        return HealthVerdict.FAILED, ("no_runs",)
    findings: list[str] = []
    baseline = runs[0]
    baseline_targets = tuple(item.target for item in baseline.suite_results)
    for run in runs:
        if run.verdict is not HealthVerdict.PASS:
            findings.append(f"run_{run.run_index}_verdict:{run.verdict.value}")
    for run in runs:
        run_targets = tuple(item.target for item in run.suite_results)
        if run_targets != baseline_targets:
            findings.append(
                f"target_set_variance:run_{run.run_index}:"
                f"{','.join(run_targets)}"
            )
    for target in baseline_targets:
        baseline_suite = next(item for item in baseline.suite_results if item.target == target)
        for run in runs[1:]:
            run_targets = {item.target for item in run.suite_results}
            if target not in run_targets:
                continue
            current_suite = next(item for item in run.suite_results if item.target == target)
            if current_suite.collected != baseline_suite.collected:
                findings.append(
                    f"collected_variance:{target}:run_{run.run_index}"
                )
            if current_suite.passed != baseline_suite.passed:
                findings.append(f"passed_variance:{target}:run_{run.run_index}")
            if current_suite.failed != baseline_suite.failed:
                findings.append(f"failed_variance:{target}:run_{run.run_index}")
            if current_suite.errors != baseline_suite.errors:
                findings.append(f"errors_variance:{target}:run_{run.run_index}")
            if current_suite.skipped != baseline_suite.skipped:
                findings.append(f"skipped_variance:{target}:run_{run.run_index}")
            if current_suite.xfailed != baseline_suite.xfailed:
                findings.append(f"xfailed_variance:{target}:run_{run.run_index}")
            if current_suite.xpassed != baseline_suite.xpassed:
                findings.append(f"xpassed_variance:{target}:run_{run.run_index}")
            if current_suite.verdict != baseline_suite.verdict:
                findings.append(f"outcome_variance:{target}:run_{run.run_index}")
    for run in runs[1:]:
        if run.verdict != baseline.verdict:
            findings.append(f"aggregate_verdict_variance:run_{run.run_index}")
    verdict = HealthVerdict.PASS if not findings else HealthVerdict.FAILED
    return verdict, tuple(findings)
