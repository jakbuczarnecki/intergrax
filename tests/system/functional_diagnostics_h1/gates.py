# © Artur Czarnecki. All rights reserved.

"""H1 health gate implementations."""

from __future__ import annotations

import re
from pathlib import Path

from tests.system.functional_diagnostics_h1.inventory import (
    COLLECTION_PYTEST_TARGETS,
    CORE_DETERMINISTIC_PYTEST_TARGETS,
    LOCAL_INTEGRATION_TARGETS,
    QUALIFICATION_RUNNERS,
    REPEATABILITY_PYTEST_TARGETS,
    SLOW_ARCHITECTURE_PYTEST_TARGETS,
    build_diagnostic_test_inventory,
    verify_invariant_owners,
)
from tests.system.functional_diagnostics_h1.models import (
    DiagnosticTestSuiteResult,
    ExternalDependencyState,
    GateResult,
    HealthGateId,
    HealthVerdict,
    PytestSubprocessResult,
    SkipClassification,
    SkipXfailFinding,
)
from tests.system.functional_diagnostics_h1.preflight import classify_external_dependencies
from tests.system.functional_diagnostics_h1.subprocess_pytest import (
    classify_pytest_exit,
    run_pytest_subprocess,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SKIP_RE = re.compile(r"@pytest\.mark\.(skip|skipif|xfail)\b|pytest\.(skip|xfail)\(")
_SLEEP_RE = re.compile(r"\b(time\.sleep|asyncio\.sleep)\(")
_STALE_PENDING_CLEANUP_RE = re.compile(
    r"pending without canonical.*cleanup|cleanup.*pending without canonical",
    re.IGNORECASE,
)
_FAIL_CLOSED_RE = re.compile(r"fail[-_ ]closed|consistency_pending", re.IGNORECASE)


def gate_h1_a_collection(inventory_count: int) -> tuple[GateResult, int]:
    result = run_pytest_subprocess(COLLECTION_PYTEST_TARGETS, collect_only=True, timeout_seconds=300.0)
    details: list[str] = [
        f"collected={result.collected_count}",
        f"inventory_files={inventory_count}",
        f"collection_errors={result.collection_errors}",
    ]
    verdict = HealthVerdict.PASS
    if result.collection_errors > 0 or result.exit_code not in {0, 5}:
        verdict = HealthVerdict.FAILED
        details.append("collection_errors_detected")
    if inventory_count <= 0:
        verdict = HealthVerdict.FAILED
        details.append("inventory_empty")
    if result.collected_count is None or result.collected_count <= 0:
        verdict = HealthVerdict.FAILED
        details.append("zero_tests_discovered")
    return (
        GateResult(
            gate_id=HealthGateId.H1_A_COLLECTION,
            verdict=verdict,
            summary=f"collection health: {verdict.value}",
            details=tuple(details),
        ),
        result.collected_count,
    )


def _merge_pytest_results(*results: PytestSubprocessResult) -> PytestSubprocessResult:
    if not results:
        return PytestSubprocessResult(
            exit_code=0,
            collected_count=None,
            passed=0,
            failed=0,
            skipped=0,
            xfailed=0,
            xpassed=0,
            errors=0,
            collection_errors=0,
            stdout_tail="",
            stderr_tail="",
            duration_seconds=0.0,
        )
    exit_code = 0
    for item in results:
        if item.exit_code not in {0, 5}:
            exit_code = item.exit_code
            break
    collected_values = [item.collected_count for item in results if item.collected_count is not None]
    merged_collected: int | None = sum(collected_values) if collected_values else None
    return PytestSubprocessResult(
        exit_code=exit_code,
        collected_count=merged_collected,
        passed=sum(item.passed for item in results),
        failed=sum(item.failed for item in results),
        skipped=sum(item.skipped for item in results),
        xfailed=sum(item.xfailed for item in results),
        xpassed=sum(item.xpassed for item in results),
        errors=sum(item.errors for item in results),
        collection_errors=sum(item.collection_errors for item in results),
        stdout_tail=results[-1].stdout_tail,
        stderr_tail="\n".join(item.stderr_tail for item in results if item.stderr_tail),
        duration_seconds=sum(item.duration_seconds for item in results),
    )


def gate_h1_b_core_health() -> tuple[GateResult, PytestSubprocessResult]:
    core_result = run_pytest_subprocess(CORE_DETERMINISTIC_PYTEST_TARGETS, timeout_seconds=1800.0)
    slow_result = run_pytest_subprocess(SLOW_ARCHITECTURE_PYTEST_TARGETS, timeout_seconds=1800.0)
    merged = _merge_pytest_results(core_result, slow_result)
    outcomes = (
        classify_pytest_exit(core_result),
        classify_pytest_exit(slow_result),
    )
    verdict = HealthVerdict.PASS
    if any(outcome != "PASS" for outcome in outcomes):
        verdict = HealthVerdict.FAILED
    return (
        GateResult(
            gate_id=HealthGateId.H1_B_CORE_HEALTH,
            verdict=verdict,
            summary=(
                f"core deterministic suite: passed={merged.passed} failed={merged.failed} "
                f"skipped={merged.skipped} xfailed={merged.xfailed}"
            ),
            details=outcomes,
        ),
        merged,
    )


def _validate_repeatability_metrics(result: PytestSubprocessResult) -> tuple[bool, str | None]:
    executed = result.passed + result.failed + result.skipped + result.xfailed + result.xpassed
    if result.passed > 0 and result.collected_count == 0:
        return False, "passed_gt_zero_but_collected_zero"
    if result.passed > 0 and result.collected_count is None:
        return False, "passed_gt_zero_but_collected_unknown"
    if result.collected_count is not None and executed > 0 and result.collected_count < executed:
        return False, "collected_lt_executed"
    return True, None


def gate_h1_c_repeatability() -> tuple[GateResult, tuple[DiagnosticTestSuiteResult, ...]]:
    runs: list[DiagnosticTestSuiteResult] = []
    metric_violations: list[str] = []
    for index in range(3):
        result = run_pytest_subprocess(REPEATABILITY_PYTEST_TARGETS, timeout_seconds=1200.0)
        outcome = classify_pytest_exit(result)
        metrics_ok, metric_note = _validate_repeatability_metrics(result)
        if not metrics_ok and metric_note is not None:
            metric_violations.append(f"run_{index + 1}:{metric_note}")
        verdict = HealthVerdict.PASS if outcome == "PASS" and metrics_ok else HealthVerdict.FAILED
        runs.append(
            DiagnosticTestSuiteResult(
                scope=f"repeatability_run_{index + 1}",
                collected=result.collected_count,
                passed=result.passed,
                failed=result.failed,
                skipped=result.skipped,
                xfailed=result.xfailed,
                verdict=verdict,
            )
        )
    counts = {run.collected for run in runs}
    verdicts = {run.verdict for run in runs}
    gate_verdict = HealthVerdict.PASS
    details: list[str] = []
    if len(counts) != 1:
        gate_verdict = HealthVerdict.FAILED
        details.append("collected_count_variance")
    if verdicts != {HealthVerdict.PASS}:
        gate_verdict = HealthVerdict.FAILED
        details.append("outcome_variance")
    if metric_violations:
        gate_verdict = HealthVerdict.FAILED
        details.extend(metric_violations)
    return (
        GateResult(
            gate_id=HealthGateId.H1_C_REPEATABILITY,
            verdict=gate_verdict,
            summary=f"repeatability over 3 runs: {gate_verdict.value}",
            details=tuple(details),
        ),
        tuple(runs),
    )


def gate_h1_d_invariant_coverage() -> GateResult:
    missing = verify_invariant_owners(_REPO_ROOT)
    verdict = HealthVerdict.PASS if not missing else HealthVerdict.FAILED
    return GateResult(
        gate_id=HealthGateId.H1_D_INVARIANT_COVERAGE,
        verdict=verdict,
        summary=f"invariant owners: missing={len(missing)}",
        details=missing,
    )


def _scan_skip_xfail(paths: tuple[Path, ...]) -> tuple[SkipXfailFinding, ...]:
    findings: list[SkipXfailFinding] = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            match = _SKIP_RE.search(line)
            if match is None:
                continue
            marker = match.group(1) or match.group(2)
            classification = SkipClassification.JUSTIFIED
            note = "explicit marker"
            if "no_ci" in line:
                note = "no_ci marker"
            if "TODO" in line or "FIXME" in line:
                classification = SkipClassification.STALE
                note = "todo/fixme near skip"
            findings.append(
                SkipXfailFinding(
                    path=path.relative_to(_REPO_ROOT).as_posix(),
                    line=line_no,
                    marker=marker,
                    classification=classification,
                    note=note,
                )
            )
    return tuple(findings)


def gate_h1_e_skip_xfail_honesty() -> tuple[GateResult, tuple[SkipXfailFinding, ...]]:
    inventory = build_diagnostic_test_inventory()
    paths = tuple(_REPO_ROOT / item.path for item in inventory)
    findings = _scan_skip_xfail(paths)
    stale = tuple(item for item in findings if item.classification is SkipClassification.STALE)
    masking = tuple(item for item in findings if item.classification is SkipClassification.MASKING_FAILURE)
    verdict = HealthVerdict.PASS
    details: list[str] = [f"skip_xfail_count={len(findings)}"]
    if stale or masking:
        verdict = HealthVerdict.FAILED
        details.append(f"stale={len(stale)} masking={len(masking)}")
    return (
        GateResult(
            gate_id=HealthGateId.H1_E_SKIP_XFAIL_HONESTY,
            verdict=verdict,
            summary=f"skip/xfail audit: {verdict.value}",
            details=tuple(details),
        ),
        findings,
    )


def gate_h1_f_external_dependency() -> GateResult:
    statuses = classify_external_dependencies()
    failed = [
        item
        for item in statuses
        if item.state is ExternalDependencyState.FAILED_PREFLIGHT
    ]
    verdict = HealthVerdict.PASS if not failed else HealthVerdict.FAILED
    details = tuple(f"{item.family.value}:{item.state.value}" for item in statuses)
    return GateResult(
        gate_id=HealthGateId.H1_F_EXTERNAL_DEPENDENCY,
        verdict=verdict,
        summary="external dependency classification explicit",
        details=details,
    )


def gate_h1_g_runner_integrity() -> GateResult:
    missing: list[str] = []
    for runner in QUALIFICATION_RUNNERS:
        runner_path = _REPO_ROOT / runner.runner_path
        if not runner_path.exists():
            missing.append(f"missing_runner:{runner.family.value}:{runner.runner_path}")
        if runner.doc_path is not None and not (_REPO_ROOT / runner.doc_path).exists():
            missing.append(f"missing_doc:{runner.family.value}:{runner.doc_path}")
        if runner.powershell_path is not None and not (_REPO_ROOT / runner.powershell_path).exists():
            missing.append(f"missing_ps1:{runner.family.value}:{runner.powershell_path}")
    verdict = HealthVerdict.PASS if not missing else HealthVerdict.FAILED
    return GateResult(
        gate_id=HealthGateId.H1_G_RUNNER_INTEGRITY,
        verdict=verdict,
        summary=f"qualification runner integrity: missing={len(missing)}",
        details=tuple(missing),
    )


def gate_h1_h_stale_dead() -> tuple[GateResult, tuple[str, ...]]:
    stale_findings: list[str] = []
    inventory = build_diagnostic_test_inventory()
    for item in inventory:
        if not (_REPO_ROOT / item.path).exists():
            stale_findings.append(f"inventory_stale_path:{item.path}")
    verdict = HealthVerdict.PASS if not stale_findings else HealthVerdict.FAILED
    return (
        GateResult(
            gate_id=HealthGateId.H1_H_STALE_DEAD,
            verdict=verdict,
            summary=f"stale/dead detection: findings={len(stale_findings)}",
            details=tuple(stale_findings),
        ),
        tuple(stale_findings),
    )


def gate_h1_i_supersession() -> GateResult:
    r1_paths = (
        _REPO_ROOT / "tests/unit/runtime/diagnostics/test_diag_functional_read_r1_r3_active_writer_safety.py",
        _REPO_ROOT / "tests/system/functional_diagnostics_read_r1_r3/mongo_active_writer_qualification.py",
        _REPO_ROOT / "tests/system/functional_diagnostics_read_r1_r2/mongo_recovery_qualification.py",
    )
    contradictions: list[str] = []
    for path in r1_paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if _STALE_PENDING_CLEANUP_RE.search(text) and not _FAIL_CLOSED_RE.search(text):
            contradictions.append(f"stale_cleanup_semantics:{path.relative_to(_REPO_ROOT).as_posix()}")
    verdict = HealthVerdict.PASS if not contradictions else HealthVerdict.FAILED
    return GateResult(
        gate_id=HealthGateId.H1_I_SUPERSESSION,
        verdict=verdict,
        summary=f"supersession consistency: contradictions={len(contradictions)}",
        details=tuple(contradictions),
    )


def gate_h1_k_local_integration() -> GateResult:
    """H1-K: local diagnostic integration — deterministic, must PASS."""
    env_overrides = {
        "INTERGRAX_DIAGNOSTIC_PROBLEM_LIST_CURSOR_SECRET": (
            "h1-local-integration-diagnostic-cursor-secret"
        ),
    }
    suite_details: list[str] = []
    any_failed = False
    for target in LOCAL_INTEGRATION_TARGETS:
        result = run_pytest_subprocess(
            (target,),
            timeout_seconds=900.0,
            env_overrides=env_overrides,
        )
        outcome = classify_pytest_exit(result)
        suite_details.append(
            f"{target}:outcome={outcome}:passed={result.passed}:failed={result.failed}:"
            f"errors={result.errors}:dependency_class=LOCAL_DETERMINISTIC"
        )
        if outcome != "PASS":
            any_failed = True
    verdict = HealthVerdict.FAILED if any_failed else HealthVerdict.PASS
    return GateResult(
        gate_id=HealthGateId.H1_K_LOCAL_INTEGRATION,
        verdict=verdict,
        summary=f"local diagnostic integration (H1-K): {verdict.value}",
        details=tuple(suite_details),
    )


def gate_local_integration() -> GateResult:
    """Backward-compatible alias for H1-K local integration gate."""
    return gate_h1_k_local_integration()


def scan_sleep_synchronization() -> tuple[str, ...]:
    inventory = build_diagnostic_test_inventory()
    findings: list[str] = []
    for item in inventory:
        path = _REPO_ROOT / item.path
        if not path.exists():
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _SLEEP_RE.search(line) and "timeout" not in line.lower():
                findings.append(f"{item.path}:{line_no}:{line.strip()}")
    return tuple(findings)
