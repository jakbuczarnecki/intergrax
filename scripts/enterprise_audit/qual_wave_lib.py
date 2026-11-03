#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""ENT-AUDIT-X qualification wave library (testable orchestration primitives)."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from intergrax.platform.enterprise_audit.baseline import candidate_baseline_path
from intergrax.platform.enterprise_audit.contracts import AuditMode, EnterpriseAuditExitCode
from intergrax.platform.enterprise_audit.manifest import load_default_manifest
from intergrax.platform.enterprise_audit.runner import EnterpriseAuditRunner

_CLI = Path(__file__).resolve().parent / "run_enterprise_audit.py"

_SEMANTIC_CHECK_COUNT = 85
_ENT_A_SEMANTIC_COUNT = 63
_ENT_A_ACCEPTANCE_GLOB = "tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_*.py"
_TEN_ACCEPTANCE_GLOB = "tests/qualification/ent_audit_x/acceptance/test_ent_a_ten_*.py"
_VBN_ACCEPTANCE_GLOB = "tests/qualification/ent_audit_x/acceptance/test_ent_a_vbn_*.py"
_ADV_PATHS = (
    "tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py",
    "tests/qualification/ent_audit_x/unit/test_report_status_mutation.py",
)
_D31_PATHS = ("tests/qualification/ent_audit_x/unit/test_vbn_domain_validator.py",)

_SUMMARY_LINE = re.compile(r"(?:(\d+) failed, )?(?:(\d+) passed)(?:, (\d+) skipped)?(?:, (\d+) error)?.* in [\d.]+s")


@dataclass
class StageResult:
    name: str
    exit_code: int
    duration_ms: int
    skipped: bool = False
    detail: str = ""


@dataclass
class QualWaveReport:
    profile: str
    qualification_sha: str
    pinned_sha: str | None
    clean_tree: bool
    certifying: bool
    stages: list[StageResult] = field(default_factory=list)
    status: str = "FAIL"
    error: str = ""
    interruption_class: str = ""


@dataclass
class PytestSummary:
    passed: int = 0
    skipped: int = 0
    failed: int = 0
    errors: int = 0


def parse_pytest_summary(output: str) -> PytestSummary:
    summary = PytestSummary()
    for line in output.splitlines():
        match = _SUMMARY_LINE.search(line.strip())
        if not match:
            continue
        failed, passed, skipped, errors = match.groups()
        summary.failed = int(failed or 0)
        summary.passed = int(passed or 0)
        summary.skipped = int(skipped or 0)
        summary.errors = int(errors or 0)
    return summary


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_clean(root: Path) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return out.strip() == ""


def validate_certifying_pin(root: Path, pinned_sha: str | None, *, certifying: bool) -> tuple[bool, str]:
    if not certifying:
        return True, ""
    if not pinned_sha:
        return False, "TEST_EXECUTION_SHA required for certifying profile"
    head = git_head(root)
    if pinned_sha != head:
        return False, f"TEST_EXECUTION_SHA {pinned_sha} != HEAD {head}"
    if not re.fullmatch(r"[0-9a-f]{40}", pinned_sha):
        return False, "TEST_EXECUTION_SHA must be 40-char hex"
    return True, ""


def pytest_collect_count(
    root: Path,
    *,
    marker: str | None,
    paths: list[str],
    env: dict[str, str] | None = None,
) -> int:
    cmd = ["uv", "run", "pytest", *paths, "--collect-only", "-q"]
    if marker:
        cmd.extend(["-m", marker])
    merged = os.environ.copy()
    if env:
        merged.update(env)
    completed = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=False, env=merged)
    text = completed.stdout + completed.stderr
    match = re.search(r"(\d+) tests? collected", text)
    if match:
        return int(match.group(1))
    if "no tests collected" in text.lower():
        return 0
    raise RuntimeError(f"pytest collect failed: {text[-500:]}")


def run_pytest_strict(
    root: Path,
    *,
    marker: str | None,
    paths: list[str],
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
    expected_passed: int | None = None,
    allow_skipped: bool = False,
) -> tuple[int, str, PytestSummary]:
    cmd = ["uv", "run", "pytest", *paths, "-q", "--tb=line", "--no-header"]
    if marker:
        cmd.extend(["-m", marker])
    merged = os.environ.copy()
    if env:
        merged.update(env)
    try:
        completed = subprocess.run(
            cmd,
            cwd=root,
            check=False,
            env=merged,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        return 124, "pytest timeout", PytestSummary()
    summary = parse_pytest_summary(completed.stdout + completed.stderr)
    if completed.returncode != 0:
        msg = (
            f"pytest exit {completed.returncode} passed={summary.passed} "
            f"skipped={summary.skipped} failed={summary.failed}"
        )
        return (int(completed.returncode), msg, summary)
    if not allow_skipped and (summary.skipped or summary.failed or summary.errors):
        return (
            1,
            f"non-zero skips/failures skipped={summary.skipped} failed={summary.failed} errors={summary.errors}",
            summary,
        )
    if expected_passed is not None and summary.passed != expected_passed:
        return (
            1,
            f"passed {summary.passed} != expected {expected_passed}",
            summary,
        )
    return 0, f"passed={summary.passed}", summary


def run_cli_full(root: Path, output_dir: Path, *, timeout_s: int = 7200) -> tuple[int, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(_CLI),
        "run",
        "--mode",
        "FULL",
        "--output-dir",
        str(output_dir),
        "--evidence-bundle",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(cmd, cwd=root, check=False, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return 124, "L3 CLI FULL timeout"
    detail = f"cli_full exit={completed.returncode} ms={int((time.perf_counter() - started) * 1000)}"
    return int(completed.returncode), detail


def validate_full_artifacts(output_dir: Path, expected_sha: str) -> tuple[bool, str]:
    report_path = output_dir / "enterprise_audit_report.json"
    if not report_path.is_file():
        return False, "missing enterprise_audit_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    em = payload.get("execution_manifest") or {}
    if payload.get("current_sha") != expected_sha:
        return False, f"report current_sha != {expected_sha}"
    if em.get("qualification_sha") != expected_sha:
        return False, "manifest qualification_sha mismatch"
    if em.get("checks_planned") != 85 or em.get("checks_executed") != 85:
        return False, "checks_planned/executed not 85/85"
    if em.get("checks_skipped", 1) != 0:
        return False, "checks_skipped must be 0"
    if em.get("coverage_label") != "FULL":
        return False, "coverage_label not FULL"
    if payload.get("exit_code") != 0:
        return False, f"exit_code {payload.get('exit_code')}"
    if not payload.get("certifying_run"):
        return False, "certifying_run false"
    from intergrax.platform.enterprise_audit.validators import (
        validate_all_artifacts_in_dir,
        validate_report_semantic_consistency,
    )

    errors = validate_all_artifacts_in_dir(output_dir)
    errors += validate_report_semantic_consistency(report_path)
    if errors:
        return False, "; ".join(errors[:3])
    evidence_path = output_dir / "evidence_manifest.v1.json"
    if not evidence_path.is_file():
        return False, "missing evidence_manifest.v1.json (closed-world bundle)"
    from intergrax.platform.enterprise_audit.evidence import verify_evidence_manifest

    if not verify_evidence_manifest(output_dir):
        return False, "evidence_manifest verification failed"
    return True, "FULL 85/85 + evidence integrity validated"


def run_first_freeze_candidate(root: Path, wave_output: Path) -> tuple[int, str]:
    baseline_dir = wave_output / "freeze-baseline"
    out_dir = wave_output / "freeze-candidate-run"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    runner = EnterpriseAuditRunner(root, dry_run=False, require_clean_repo=True)
    _, manifest, code = runner.run(
        AuditMode.FREEZE,
        output_dir=out_dir,
        baseline_dir=baseline_dir,
        bootstrap_baseline=True,
        certifying_run=True,
    )
    if code is not EnterpriseAuditExitCode.CLEAN:
        return int(code), f"FREEZE candidate exit {code}"
    if manifest.checks_executed != 85 or manifest.checks_skipped != 0:
        return 1, f"FREEZE candidate {manifest.checks_executed}/85 skipped={manifest.checks_skipped}"
    cand = candidate_baseline_path(baseline_dir)
    if not cand.is_file():
        return 1, "candidate baseline file missing"
    accepted = baseline_dir / "accepted_baseline.v1.json"
    if accepted.is_file():
        return 1, "accepted baseline must not exist for first candidate gate"
    return 0, str(cand.name)


def _glob_paths(root: Path, pattern: str) -> list[str]:
    rel = pattern.replace("/", os.sep)
    return sorted(str(p.relative_to(root)).replace("\\", "/") for p in root.glob(rel))


def run_qualification_wave(
    root: Path,
    *,
    profile: str,
    pinned_sha: str | None,
    l3_output: Path,
    wave_output: Path,
    run_pytest_fn: Callable[..., tuple[int, str, PytestSummary]] | None = None,
) -> QualWaveReport:
    certifying = profile == "certifying"
    head = git_head(root)
    report = QualWaveReport(
        profile=profile,
        qualification_sha=head,
        pinned_sha=pinned_sha,
        clean_tree=git_clean(root),
        certifying=certifying,
    )
    if not report.clean_tree:
        report.error = "working tree not clean (clean checkout required)"
        return report
    pin_ok, pin_msg = validate_certifying_pin(root, pinned_sha, certifying=certifying)
    if not pin_ok:
        report.error = pin_msg
        return report

    pytest_runner = run_pytest_fn or run_pytest_strict
    wave_output.mkdir(parents=True, exist_ok=True)

    acceptance_paths = _glob_paths(root, _ENT_A_ACCEPTANCE_GLOB)
    semantic_env = {"ENT_AUDIT_X_SEMANTIC": "1"}
    matrix_env = {"ENT_AUDIT_X_GATE_MATRIX": "1"}

    try:
        acceptance_expected = pytest_collect_count(
            root,
            marker="ent_semantic_gate",
            paths=acceptance_paths,
            env=semantic_env,
        )
        if acceptance_expected != _ENT_A_SEMANTIC_COUNT:
            report.error = f"ENT-A acceptance collect {acceptance_expected} != {_ENT_A_SEMANTIC_COUNT}"
            return report
        matrix_expected = _SEMANTIC_CHECK_COUNT
    except RuntimeError as exc:
        report.error = str(exc)
        return report

    stages: list[tuple[str, Callable[[], tuple[int, str]]]] = [
        (
            "l1",
            lambda: pytest_runner(
                root,
                marker="ent_l1",
                paths=["tests/qualification/ent_audit_x"],
            )[:2],
        ),
        (
            "l2",
            lambda: pytest_runner(
                root,
                marker="ent_l2",
                paths=["tests/qualification/ent_audit_x/integration"],
            )[:2],
        ),
        (
            "semantic_acceptance",
            lambda: pytest_runner(
                root,
                marker="ent_semantic_gate",
                paths=acceptance_paths,
                env=semantic_env,
                timeout_s=3600,
                expected_passed=acceptance_expected,
            )[:2],
        ),
        (
            "semantic_matrix",
            lambda: pytest_runner(
                root,
                marker="ent_semantic_matrix",
                paths=["tests/qualification/ent_audit_x/integration/test_semantic_gate_execution.py"],
                env=matrix_env,
                timeout_s=7200,
                expected_passed=matrix_expected,
            )[:2],
        ),
        (
            "adv31",
            lambda: pytest_runner(
                root,
                marker=None,
                paths=list(_ADV_PATHS),
                timeout_s=1800,
                expected_passed=pytest_collect_count(root, marker=None, paths=list(_ADV_PATHS)),
            )[:2],
        ),
        (
            "ten13",
            lambda: pytest_runner(
                root,
                marker="ent_semantic_gate",
                paths=_glob_paths(root, _TEN_ACCEPTANCE_GLOB),
                env=semantic_env,
                timeout_s=1800,
                expected_passed=13,
            )[:2],
        ),
        (
            "vbn9",
            lambda: pytest_runner(
                root,
                marker="ent_semantic_gate",
                paths=_glob_paths(root, _VBN_ACCEPTANCE_GLOB),
                env=semantic_env,
                timeout_s=1800,
                expected_passed=9,
            )[:2],
        ),
        (
            "d31",
            lambda: pytest_runner(
                root,
                marker=None,
                paths=list(_D31_PATHS),
                timeout_s=900,
            )[:2],
        ),
    ]

    for name, fn in stages:
        started = time.perf_counter()
        try:
            code, detail = fn()
        except subprocess.TimeoutExpired:
            code, detail = 124, "timeout"
            report.interruption_class = "TIMEOUT"
        dur = int((time.perf_counter() - started) * 1000)
        report.stages.append(StageResult(name=name, exit_code=code, duration_ms=dur, detail=detail))
        if code != 0:
            if code == 124:
                report.interruption_class = report.interruption_class or "TIMEOUT"
            report.error = f"stage {name} failed: {detail}"
            return report

    planned = load_default_manifest()
    if len(planned) != _SEMANTIC_CHECK_COUNT:
        report.error = f"manifest CheckID count {len(planned)} != {_SEMANTIC_CHECK_COUNT}"
        return report

    if certifying:
        started = time.perf_counter()
        code, detail = run_cli_full(root, l3_output)
        dur = int((time.perf_counter() - started) * 1000)
        report.stages.append(StageResult(name="l3_cli_full", exit_code=code, duration_ms=dur, detail=detail))
        if code != 0:
            report.error = f"L3 CLI FULL failed: {detail}"
            if code == 124:
                report.interruption_class = "TIMEOUT"
            return report
        ok, msg = validate_full_artifacts(l3_output, head)
        report.stages.append(
            StageResult(name="l3_validate", exit_code=0 if ok else 1, duration_ms=0, detail=msg),
        )
        if not ok:
            report.error = f"L3 artifact validation: {msg}"
            return report

        started = time.perf_counter()
        code, detail = run_first_freeze_candidate(root, wave_output)
        dur = int((time.perf_counter() - started) * 1000)
        report.stages.append(
            StageResult(name="first_freeze_candidate", exit_code=code, duration_ms=dur, detail=detail),
        )
        if code != 0:
            report.error = f"first FREEZE candidate failed: {detail}"
            return report
    else:
        report.stages.append(
            StageResult(
                name="l3_cli_full",
                exit_code=0,
                duration_ms=0,
                skipped=True,
                detail="pre-qual profile (not certifying)",
            ),
        )
        report.stages.append(
            StageResult(
                name="first_freeze_candidate",
                exit_code=0,
                duration_ms=0,
                skipped=True,
                detail="pre-qual profile (not certifying)",
            ),
        )

    report.status = "PASS" if certifying else "PRE_QUAL_PASS"
    return report


def write_report(report: QualWaveReport, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "qual_wave_report.json"
    payload: dict[str, Any] = {
        **asdict(report),
        "stages": [asdict(s) for s in report.stages],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
