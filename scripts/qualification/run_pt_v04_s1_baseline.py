#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""PT-V04 S1 diagnostic baseline runner (non-certifying; W2-RECOVERY-02)."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / ".tmp/session/w2-recovery-02/pt-v04"

PT_V04_TESTS = [
    "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py",
    "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py",
    "tests/qualification/state_x/test_state_x_p0_baseline.py",
    "tests/qualification/trace_x/test_trace_x_p0_baseline.py",
    "tests/qualification/trace_x/test_trace_x_p6_closed_world_gates.py",
    "tests/qualification/tenant_x/test_tenant_x_p0_fast.py",
]


def _sha(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def _worktree_clean(cwd: Path) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True)
    return out.strip() == ""


def _tracked_tree_unchanged(cwd: Path) -> bool:
    unstaged = subprocess.run(["git", "diff", "--quiet"], cwd=cwd).returncode == 0
    staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=cwd).returncode == 0
    return unstaged and staged


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_step(cmd: list[str], label: str, evidence_dir: Path) -> dict[str, object]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = evidence_dir / f"{label}.stdout.txt"
    stderr_path = evidence_dir / f"{label}.stderr.txt"
    try:
        proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, shell=False, check=False)
    except FileNotFoundError as exc:
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text(str(exc), encoding="utf-8")
        return {
            "label": label,
            "command_argv": cmd,
            "exit_code": 127,
            "complete": False,
            "stdout_path": str(stdout_path.relative_to(_REPO)),
            "stderr_path": str(stderr_path.relative_to(_REPO)),
            "stdout_sha256": _sha256_file(stdout_path),
            "stderr_sha256": _sha256_file(stderr_path),
        }

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    return {
        "label": label,
        "command_argv": cmd,
        "exit_code": proc.returncode,
        "complete": proc.returncode == 0,
        "stdout_path": str(stdout_path.relative_to(_REPO)),
        "stderr_path": str(stderr_path.relative_to(_REPO)),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
    }


def main() -> int:
    head_at_entry = _sha(_REPO)
    if not _worktree_clean(_REPO):
        summary = {
            "execution_sha": head_at_entry,
            "recorded_at": datetime.now(UTC).isoformat(),
            "node": "PT-V04",
            "certifying": False,
            "complete": False,
            "error": "dirty tracked worktree at entry",
            "tracked_tree_unchanged": False,
            "results": [],
        }
        out_json = _OUT / head_at_entry[:12] / "PT_V04_S1_BASELINE.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 1

    evidence_base = _OUT / head_at_entry[:12]
    results: list[dict[str, object]] = []

    f10_check_argv = [sys.executable, "scripts/qualification/render_ebh7_f10_normative_matrix.py", "--check"]
    results.append(_run_step(f10_check_argv, "F10_check", evidence_base))

    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", *PT_V04_TESTS]
    results.append(_run_step(pytest_cmd, "PT_V04_pytest_suite", evidence_base))

    trace_argv = [sys.executable, "scripts/governance/verify_arch_freeze_plan_traceability.py"]
    results.append(_run_step(trace_argv, "traceability", evidence_base))

    head_at_exit = _sha(_REPO)
    tree_ok = _tracked_tree_unchanged(_REPO)
    head_unchanged = head_at_entry == head_at_exit
    subprocess_ok = all(r.get("exit_code") == 0 for r in results)
    complete = subprocess_ok and tree_ok and head_unchanged

    summary = {
        "execution_sha": head_at_entry,
        "recorded_at": datetime.now(UTC).isoformat(),
        "node": "PT-V04",
        "certifying": False,
        "results": results,
        "complete": complete,
        "tracked_tree_unchanged": tree_ok,
        "head_unchanged": head_unchanged,
    }
    if not head_unchanged:
        summary["error"] = "HEAD changed during qualification"
    if not tree_ok:
        summary["error"] = (summary.get("error") or "") + " tracked tree mutated during qualification"

    out_json = evidence_base / "PT_V04_S1_BASELINE.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"complete": complete, "path": str(out_json)}, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
