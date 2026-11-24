#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Campaign B-1/B-2 alpha runner (diagnostic; W2-RECOVERY-02)."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).with_name("campaign_b_w2_manifest.v1.json")
_OUT_ROOT = _REPO / ".tmp/session/w2-recovery-02/campaign-b"

OWNER_BY_PHASE = {"b1": "PT-B01", "b2": "PT-B02"}


@dataclass
class BaselineValidation:
    ok: bool
    errors: list[str]
    execution_sha: str | None = None
    cwd: Path | None = None


def _git_head(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def _git_clean(cwd: Path) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True)
    return out.strip() == ""


def _parse_command(command: str) -> list[str]:
    parts = command.split()
    if not parts:
        raise ValueError("empty command")
    if parts[0] == "python":
        return [sys.executable, *parts[1:]]
    return parts


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command_targets_argv(argv: list[str]) -> list[Path]:
    paths: list[Path] = []
    for token in argv[1:]:
        if token.startswith("-"):
            continue
        if "/" in token or token.endswith(".py"):
            paths.append(Path(token))
    return paths


def validate_b1_baseline(worktree: Path | None, baseline_sha: str) -> BaselineValidation:
    errors: list[str] = []
    if worktree is None:
        return BaselineValidation(False, ["b1 requires --worktree"])
    if not worktree.is_dir():
        return BaselineValidation(False, [f"worktree not found: {worktree}"])
    try:
        head = _git_head(worktree)
    except subprocess.CalledProcessError:
        return BaselineValidation(False, [f"not a git worktree: {worktree}"])
    if head != baseline_sha:
        errors.append(f"HEAD {head} != b1_baseline_sha {baseline_sha}")
    if not _git_clean(worktree):
        errors.append("worktree has tracked or untracked changes")
    if errors:
        return BaselineValidation(False, errors)
    return BaselineValidation(True, [], execution_sha=head, cwd=worktree)


def validate_b2_cwd(cwd: Path) -> BaselineValidation:
    errors: list[str] = []
    try:
        head = _git_head(cwd)
    except subprocess.CalledProcessError:
        return BaselineValidation(False, ["current checkout is not a git repository"])
    if not _git_clean(cwd):
        errors.append("execution checkout is dirty")
    if errors:
        return BaselineValidation(False, errors, execution_sha=head, cwd=cwd)
    return BaselineValidation(True, [], execution_sha=head, cwd=cwd)


def _validate_manifest_cases(cases: list[dict]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for case in cases:
        for field in ("case_id", "contract", "command"):
            if not case.get(field):
                errors.append(f"malformed case missing {field}: {case}")
        cid = str(case.get("case_id", ""))
        if cid in seen:
            errors.append(f"duplicate case_id {cid}")
        seen.add(cid)
    return errors


def _artifact_available(cwd: Path, argv: list[str]) -> tuple[bool, str | None]:
    for rel in _command_targets_argv(argv):
        candidate = cwd / rel
        if not candidate.exists():
            return False, str(rel)
    return True, None


def run_case(
    case: dict[str, object],
    *,
    phase: str,
    cwd: Path,
    execution_sha: str,
    dry_run: bool = False,
) -> dict[str, object]:
    owner = OWNER_BY_PHASE[phase]
    case_id = str(case["case_id"])
    try:
        argv = _parse_command(str(case["command"]))
    except ValueError as exc:
        return {
            "case_id": case_id,
            "phase": phase,
            "contract": case.get("contract"),
            "owner": owner,
            "command_argv": [],
            "execution_sha": execution_sha,
            "exit_code": 127,
            "result": "ERROR",
            "error": str(exc),
        }

    available, missing = _artifact_available(cwd, argv)
    if not available:
        return {
            "case_id": case_id,
            "phase": phase,
            "contract": case.get("contract"),
            "owner": owner,
            "command_argv": argv,
            "execution_sha": execution_sha,
            "exit_code": None,
            "result": "NOT_EXECUTABLE_AT_BASELINE",
            "required_artifact": missing,
        }

    if dry_run:
        return {
            "case_id": case_id,
            "phase": phase,
            "owner": owner,
            "command_argv": argv,
            "execution_sha": execution_sha,
            "result": "SKIPPED",
        }

    evidence_dir = _OUT_ROOT / phase / execution_sha[:12] / case_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = evidence_dir / "stdout.txt"
    stderr_path = evidence_dir / "stderr.txt"

    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, shell=False, check=False)
    except FileNotFoundError as exc:
        return {
            "case_id": case_id,
            "phase": phase,
            "contract": case.get("contract"),
            "owner": owner,
            "command_argv": argv,
            "execution_sha": execution_sha,
            "exit_code": 127,
            "result": "EXECUTION_ERROR",
            "error": str(exc),
            "stdout_path": None,
            "stderr_path": None,
        }

    stdout_path.write_text(proc.stdout or "", encoding="utf-8")
    stderr_path.write_text(proc.stderr or "", encoding="utf-8")
    passed = proc.returncode == 0
    return {
        "case_id": case_id,
        "phase": phase,
        "contract": case.get("contract"),
        "owner": owner,
        "command_argv": argv,
        "execution_sha": execution_sha,
        "exit_code": proc.returncode,
        "result": "PASS" if passed else "FAIL",
        "stdout_path": str(stdout_path.relative_to(_REPO)),
        "stderr_path": str(stderr_path.relative_to(_REPO)),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("b1", "b2"), required=True)
    parser.add_argument("--worktree", type=Path, default=None, help="detached worktree @ B-1 baseline SHA")
    args = parser.parse_args(argv)

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["b1_cases"] if args.phase == "b1" else manifest["b2_cases"]
    manifest_errors = _validate_manifest_cases(cases)
    if manifest_errors:
        for err in manifest_errors:
            print(f"[FAIL] manifest: {err}")
        return 1

    baseline_sha = str(manifest.get("b1_baseline_sha", ""))
    if args.phase == "b1":
        gate = validate_b1_baseline(args.worktree, baseline_sha)
    else:
        gate = validate_b2_cwd(_REPO)

    if not gate.ok:
        for err in gate.errors:
            print(f"[FAIL] baseline: {err}")
        out_path = _OUT_ROOT / f"RESULTS_{args.phase.upper()}_BASELINE_REJECTED.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "phase": args.phase,
                    "recorded_at": datetime.now(UTC).isoformat(),
                    "baseline_valid": False,
                    "errors": gate.errors,
                    "rows": [],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 1

    cwd = gate.cwd or _REPO
    execution_sha = gate.execution_sha or _git_head(cwd)
    rows = [run_case(c, phase=args.phase, cwd=cwd, execution_sha=execution_sha) for c in cases]

    out = {
        "phase": args.phase,
        "recorded_at": datetime.now(UTC).isoformat(),
        "cwd": str(cwd),
        "execution_sha": execution_sha,
        "b1_baseline_sha": baseline_sha if args.phase == "b1" else None,
        "historical_replay": args.phase == "b1",
        "rows": rows,
    }
    out_path = _OUT_ROOT / execution_sha[:12] / f"RESULTS_{args.phase.upper()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['case_id']} | {row.get('contract')} | owner={row['owner']} | "
            f"exit={row.get('exit_code')} | {row['result']} | SHA={execution_sha[:12]}"
        )

    incomplete = sum(1 for r in rows if r["result"] != "PASS")
    print(f"[{'OK' if incomplete == 0 else 'PARTIAL'}] {out_path} incomplete={incomplete}/{len(rows)}")
    return 0 if incomplete == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
