#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""L1 → L2 → semantic 85 → L3 certifying orchestrator (ENT-AUDIT-X C051)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

_CLI = Path(__file__).resolve().parent / "run_enterprise_audit.py"
_DEFAULT_LOG_ROOT = Path(".tmp/session/ent-audit-x-qual-wave")


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


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_clean(root: Path) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return out.strip() == ""


def _run_pytest(
    root: Path,
    *,
    marker: str | None,
    paths: list[str],
    env: dict[str, str] | None = None,
    timeout_s: int | None = None,
) -> int:
    cmd = ["uv", "run", "pytest", *paths, "-q", "--tb=line", "--no-header"]
    if marker:
        cmd.extend(["-m", marker])
    merged = os.environ.copy()
    if env:
        merged.update(env)
    completed = subprocess.run(
        cmd,
        cwd=root,
        check=False,
        env=merged,
        timeout=timeout_s,
    )
    return int(completed.returncode)


def _run_cli_full(root: Path, output_dir: Path, *, timeout_s: int = 7200) -> tuple[int, str]:
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


def _validate_full_artifacts(output_dir: Path, expected_sha: str) -> tuple[bool, str]:
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
    return True, "FULL 85/85 validated"


def run_pre_qual_stages(root: Path, *, run_l3_cli: bool, l3_output: Path) -> QualWaveReport:
    head = git_head(root)
    report = QualWaveReport(
        profile="certifying" if run_l3_cli else "pre-qual",
        qualification_sha=head,
        pinned_sha=head,
        clean_tree=git_clean(root),
        certifying=run_l3_cli,
    )
    if not report.clean_tree:
        report.error = "working tree not clean"
        return report

    stages: list[tuple[str, callable]] = [
        ("l1", lambda: _run_pytest(root, marker="ent_l1", paths=["tests/qualification/ent_audit_x"])),
        (
            "l2",
            lambda: _run_pytest(
                root,
                marker="ent_l2",
                paths=["tests/qualification/ent_audit_x/integration"],
            ),
        ),
        (
            "semantic_acceptance",
            lambda: _run_pytest(
                root,
                marker="ent_semantic_gate",
                paths=["tests/qualification/ent_audit_x/acceptance"],
                env={"ENT_AUDIT_X_SEMANTIC": "1"},
                timeout_s=3600,
            ),
        ),
        (
            "semantic_matrix",
            lambda: _run_pytest(
                root,
                marker="ent_semantic_matrix",
                paths=[
                    "tests/qualification/ent_audit_x/integration/test_semantic_gate_execution.py",
                ],
                env={"ENT_AUDIT_X_GATE_MATRIX": "1"},
                timeout_s=7200,
            ),
        ),
    ]

    for name, fn in stages:
        started = time.perf_counter()
        try:
            code = fn()
        except subprocess.TimeoutExpired:
            code = 124
        dur = int((time.perf_counter() - started) * 1000)
        report.stages.append(StageResult(name=name, exit_code=code, duration_ms=dur))
        if code != 0:
            report.error = f"stage {name} failed exit {code}"
            return report

    try:
        from intergrax.platform.enterprise_audit.manifest import load_default_manifest

        planned = load_default_manifest()
        if len(planned) != 85:
            report.error = f"manifest CheckID count {len(planned)} != 85"
            return report
    except ValueError as exc:
        report.error = f"manifest validation: {exc}"
        return report

    if run_l3_cli:
        started = time.perf_counter()
        code, detail = _run_cli_full(root, l3_output)
        dur = int((time.perf_counter() - started) * 1000)
        report.stages.append(StageResult(name="l3_cli_full", exit_code=code, duration_ms=dur, detail=detail))
        if code != 0:
            report.error = f"L3 CLI FULL failed: {detail}"
            return report
        ok, msg = _validate_full_artifacts(l3_output, head)
        if not ok:
            report.error = f"L3 artifact validation: {msg}"
            report.stages.append(
                StageResult(name="l3_validate", exit_code=1, duration_ms=0, detail=msg),
            )
            return report
        report.stages.append(StageResult(name="l3_validate", exit_code=0, duration_ms=0, detail=msg))
    else:
        report.stages.append(
            StageResult(
                name="l3_cli_full",
                exit_code=0,
                duration_ms=0,
                skipped=True,
                detail="not requested (pre-qual profile)",
            ),
        )

    report.status = "PASS" if run_l3_cli else "PRE_QUAL_PASS"
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


def main() -> int:
    parser = argparse.ArgumentParser(description="ENT-AUDIT-X qualification wave orchestrator")
    parser.add_argument(
        "--profile",
        choices=("pre-qual", "certifying"),
        default="pre-qual",
        help="pre-qual runs L1/L2/semantic; certifying adds mandatory L3 CLI FULL 85/85",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned stages only")
    parser.add_argument("--log-dir", type=Path, default=_DEFAULT_LOG_ROOT)
    parser.add_argument("--l3-output", type=Path, default=None)
    parser.add_argument(
        "--inject-fail-after",
        choices=("l1", "l2", "semantic_acceptance", "semantic_matrix"),
        help="Self-test hook: exit 2 after named stage",
    )
    args = parser.parse_args()
    root = repo_root()

    if args.dry_run:
        print("profile=", args.profile)
        for line in (
            "l1 ent_l1",
            "l2 ent_l2 integration/",
            "semantic_acceptance ent_semantic_gate ENT_AUDIT_X_SEMANTIC=1",
            "semantic_matrix ent_semantic_matrix ENT_AUDIT_X_GATE_MATRIX=1",
            "l3_cli_full CLI FULL certifying (certifying profile only)",
        ):
            print("stage", line)
        return 0

    if args.profile == "certifying":
        pin = os.environ.get("TEST_EXECUTION_SHA")
        head = git_head(root)
        if pin and pin != head:
            print(f"TEST_EXECUTION_SHA {pin} != HEAD {head}", file=sys.stderr)
            return 1

    if args.inject_fail_after:
        # Fast self-test path without full qualification cost.
        mapping = {
            "l1": ("ent_l1", ["tests/qualification/ent_audit_x/unit/test_task_id_registry.py"]),
            "l2": ("ent_l2", ["tests/qualification/ent_audit_x/integration/test_adversarial_adv01_30.py"]),
            "semantic_acceptance": (
                "ent_semantic_gate",
                ["tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_bnd_01.py"],
            ),
            "semantic_matrix": (
                "ent_semantic_matrix",
                ["tests/qualification/ent_audit_x/integration/test_semantic_gate_execution.py"],
            ),
        }
        marker, paths = mapping[args.inject_fail_after]
        env = (
            {"ENT_AUDIT_X_SEMANTIC": "1"}
            if args.inject_fail_after == "semantic_acceptance"
            else {"ENT_AUDIT_X_GATE_MATRIX": "1"}
            if args.inject_fail_after == "semantic_matrix"
            else None
        )
        code = _run_pytest(root, marker=marker, paths=paths, env=env, timeout_s=600)
        if code == 0:
            return 2
        return 0

    l3_out = args.l3_output or (args.log_dir / "l3-cli-full")
    run_l3 = args.profile == "certifying"
    report = run_pre_qual_stages(root, run_l3_cli=run_l3, l3_output=l3_out)
    write_report(report, args.log_dir)
    if report.status not in ("PASS", "PRE_QUAL_PASS"):
        print(report.error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
