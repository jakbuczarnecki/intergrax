#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Campaign B-1/B-2 alpha runner (diagnostic; W2-RECOVERY-03)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).with_name("campaign_b_w2_manifest.v1.json")
_OUT_ROOT = _REPO / ".tmp/session/w2-recovery-03/campaign-b"

OWNER_BY_PHASE = {"b1": "PT-B01", "b2": "PT-B02"}
EXECUTION_CONTEXT_HISTORICAL = "HISTORICAL_ALPHA"
EXECUTION_CONTEXT_HEAD = "HEAD_DIAGNOSTIC"
_OUTPUT_PATH_FLAGS = frozenset({"--output-dir", "--out-dir", "-o"})
_CANONICAL_COVERAGE = {
    "b1": {"canonical_planned_cases": 18, "canonical_reserve_slots": 4},
    "b2": {"canonical_planned_cases": 9, "canonical_reserve_slots": 3},
}


@dataclass
class BaselineValidation:
    ok: bool
    errors: list[str]
    execution_sha: str | None = None
    cwd: Path | None = None
    execution_context: str = EXECUTION_CONTEXT_HISTORICAL


def _git_head(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def _git_clean(cwd: Path) -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd, text=True)
    return out.strip() == ""


def _parse_command(command: str) -> list[str]:
    parts = shlex.split(command, posix=False)
    if not parts:
        raise ValueError("empty command")
    if parts[0] == "python":
        return [sys.executable, *parts[1:]]
    return parts


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filesystem_check_path(token: str) -> Path:
    if "::" in token:
        return Path(token.split("::", 1)[0])
    return Path(token)


def _command_input_artifacts(argv: list[str]) -> list[Path]:
    paths: list[Path] = []
    skip_next = False
    for idx, token in enumerate(argv):
        if idx == 0:
            continue
        if skip_next:
            skip_next = False
            continue
        if token in _OUTPUT_PATH_FLAGS:
            skip_next = True
            continue
        if token.startswith("-") and "=" in token:
            flag = token.split("=", 1)[0]
            if flag in _OUTPUT_PATH_FLAGS:
                continue
        if token.startswith("-"):
            continue
        if "::" in token or token.endswith(".py") or token.startswith("tests/") or token.startswith("scripts/"):
            paths.append(_filesystem_check_path(token))
    return paths


def validate_historical_baseline(
    worktree: Path | None,
    baseline_sha: str,
    *,
    phase: str,
) -> BaselineValidation:
    errors: list[str] = []
    label = phase.upper()
    if worktree is None:
        return BaselineValidation(False, [f"{label} historical alpha requires --worktree"])
    if not worktree.is_dir():
        return BaselineValidation(False, [f"worktree not found: {worktree}"])
    try:
        head = _git_head(worktree)
    except subprocess.CalledProcessError:
        return BaselineValidation(False, [f"not a git worktree: {worktree}"])
    if head != baseline_sha:
        errors.append(f"HEAD {head} != {phase}_baseline_sha {baseline_sha}")
    if not _git_clean(worktree):
        errors.append("worktree has tracked or untracked changes before execution")
    if errors:
        return BaselineValidation(False, errors, execution_context=EXECUTION_CONTEXT_HISTORICAL)
    return BaselineValidation(
        True,
        [],
        execution_sha=head,
        cwd=worktree,
        execution_context=EXECUTION_CONTEXT_HISTORICAL,
    )


def validate_head_diagnostic(cwd: Path) -> BaselineValidation:
    errors: list[str] = []
    try:
        head = _git_head(cwd)
    except subprocess.CalledProcessError:
        return BaselineValidation(False, ["current checkout is not a git repository"])
    if not _git_clean(cwd):
        errors.append("HEAD_DIAGNOSTIC checkout is dirty")
    if errors:
        return BaselineValidation(
            False,
            errors,
            execution_sha=head,
            cwd=cwd,
            execution_context=EXECUTION_CONTEXT_HEAD,
        )
    return BaselineValidation(
        True,
        [],
        execution_sha=head,
        cwd=cwd,
        execution_context=EXECUTION_CONTEXT_HEAD,
    )


def validate_b1_baseline(worktree: Path | None, baseline_sha: str) -> BaselineValidation:
    return validate_historical_baseline(worktree, baseline_sha, phase="b1")


def validate_b2_baseline(worktree: Path | None, baseline_sha: str) -> BaselineValidation:
    return validate_historical_baseline(worktree, baseline_sha, phase="b2")


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
    for rel in _command_input_artifacts(argv):
        candidate = cwd / rel
        if not candidate.exists():
            return False, str(rel)
    return True, None


def _coverage_summary(phase: str, cases: list[dict], rows: list[dict]) -> dict[str, object]:
    canon = _CANONICAL_COVERAGE[phase]
    defined = len(cases)
    executed = sum(1 for r in rows if r.get("result") in ("PASS", "FAIL"))
    passed = sum(1 for r in rows if r.get("result") == "PASS")
    failed = sum(1 for r in rows if r.get("result") == "FAIL")
    incomplete = sum(1 for r in rows if r.get("result") not in ("PASS", "FAIL"))
    coverage_complete = defined >= canon["canonical_planned_cases"] and passed == defined and incomplete == 0
    defined_cases_pass = passed == defined and incomplete == 0 and defined > 0
    return {
        **canon,
        "manifest_defined_cases": defined,
        "executed_cases": executed,
        "passed_cases": passed,
        "failed_cases": failed,
        "incomplete_cases": incomplete,
        "coverage_complete": coverage_complete,
        "defined_cases_pass": defined_cases_pass,
        "campaign_alpha_complete": False,
        "coverage_classification": "PARTIAL_DIAGNOSTIC",
    }


def run_case(
    case: dict[str, object],
    *,
    phase: str,
    cwd: Path,
    execution_sha: str,
    execution_context: str,
    instance_id: str,
    dry_run: bool = False,
) -> dict[str, object]:
    owner = OWNER_BY_PHASE[phase]
    case_id = str(case["case_id"])
    base: dict[str, object] = {
        "case_id": case_id,
        "phase": phase,
        "contract": case.get("contract"),
        "owner": owner,
        "execution_sha": execution_sha,
        "execution_context": execution_context,
        "qualification_phase": "alpha" if execution_context == EXECUTION_CONTEXT_HISTORICAL else "diagnostic",
    }
    try:
        if case.get("command_argv"):
            argv = list(case["command_argv"])  # type: ignore[arg-type]
        else:
            argv = _parse_command(str(case["command"]))
    except ValueError as exc:
        return {**base, "command_argv": [], "exit_code": 127, "result": "ERROR", "error": str(exc)}

    available, missing = _artifact_available(cwd, argv)
    if not available:
        return {
            **base,
            "command_argv": argv,
            "exit_code": None,
            "result": "NOT_EXECUTABLE_AT_BASELINE",
            "required_artifact": missing,
        }

    if dry_run:
        return {**base, "command_argv": argv, "result": "SKIPPED"}

    evidence_dir = _OUT_ROOT / phase / execution_context / execution_sha[:12] / instance_id / case_id
    evidence_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = evidence_dir / "stdout.txt"
    stderr_path = evidence_dir / "stderr.txt"

    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, shell=False, check=False)
    except FileNotFoundError as exc:
        return {
            **base,
            "command_argv": argv,
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
        **base,
        "command_argv": argv,
        "exit_code": proc.returncode,
        "result": "PASS" if passed else "FAIL",
        "stdout_path": str(stdout_path.relative_to(_REPO)),
        "stderr_path": str(stderr_path.relative_to(_REPO)),
        "stdout_sha256": _sha256_file(stdout_path),
        "stderr_sha256": _sha256_file(stderr_path),
    }


def _post_execution_tree_valid(worktree: Path, baseline_sha: str) -> list[str]:
    errors: list[str] = []
    head = _git_head(worktree)
    if head != baseline_sha:
        errors.append(f"post-exec HEAD {head} != baseline {baseline_sha}")
    if not _git_clean(worktree):
        errors.append("historical worktree dirty after execution (tracked tree mutation)")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("b1", "b2"), required=True)
    parser.add_argument("--worktree", type=Path, default=None, help="detached worktree @ historical baseline SHA")
    parser.add_argument(
        "--execution-context",
        choices=(EXECUTION_CONTEXT_HISTORICAL, EXECUTION_CONTEXT_HEAD),
        default=EXECUTION_CONTEXT_HISTORICAL,
    )
    parser.add_argument("--instance-id", default="default", help="separate evidence instances under same SHA")
    args = parser.parse_args(argv)

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["b1_cases"] if args.phase == "b1" else manifest["b2_cases"]
    manifest_errors = _validate_manifest_cases(cases)
    if manifest_errors:
        for err in manifest_errors:
            print(f"[FAIL] manifest: {err}")
        return 1

    baseline_sha = str(manifest.get("b1_baseline_sha", ""))
    if args.execution_context == EXECUTION_CONTEXT_HEAD:
        gate = validate_head_diagnostic(_REPO)
    elif args.phase == "b1":
        gate = validate_b1_baseline(args.worktree, baseline_sha)
    else:
        gate = validate_b2_baseline(args.worktree, baseline_sha)

    if not gate.ok:
        for err in gate.errors:
            print(f"[FAIL] baseline: {err}")
        out_path = _OUT_ROOT / args.execution_context / f"RESULTS_{args.phase.upper()}_BASELINE_REJECTED.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "phase": args.phase,
                    "execution_context": args.execution_context,
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
    rows = [
        run_case(
            c,
            phase=args.phase,
            cwd=cwd,
            execution_sha=execution_sha,
            execution_context=gate.execution_context,
            instance_id=args.instance_id,
        )
        for c in cases
    ]

    if gate.execution_context == EXECUTION_CONTEXT_HISTORICAL and args.worktree is not None:
        post_errors = _post_execution_tree_valid(args.worktree, baseline_sha)
        if post_errors:
            for err in post_errors:
                print(f"[FAIL] post-exec integrity: {err}")
            return 1

    coverage = _coverage_summary(args.phase, cases, rows)
    out = {
        "phase": args.phase,
        "recorded_at": datetime.now(UTC).isoformat(),
        "cwd": str(cwd),
        "execution_sha": execution_sha,
        "execution_context": gate.execution_context,
        "instance_id": args.instance_id,
        "b1_baseline_sha": baseline_sha,
        "historical_replay": gate.execution_context == EXECUTION_CONTEXT_HISTORICAL,
        "campaign_alpha_complete": False,
        "coverage": coverage,
        "rows": rows,
    }
    out_path = (
        _OUT_ROOT
        / args.phase
        / gate.execution_context
        / execution_sha[:12]
        / args.instance_id
        / f"RESULTS_{args.phase.upper()}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for row in rows:
        print(
            f"{row['case_id']} | {row.get('contract')} | owner={row['owner']} | "
            f"ctx={row.get('execution_context')} | exit={row.get('exit_code')} | "
            f"{row['result']} | SHA={execution_sha[:12]}"
        )

    incomplete = sum(1 for r in rows if r["result"] != "PASS")
    label = "DEFINED_CASES_PASS" if coverage.get("defined_cases_pass") else "INCOMPLETE"
    if coverage.get("coverage_complete"):
        label = "CAMPAIGN_ALPHA_COMPLETE"
    print(
        f"[{'OK' if incomplete == 0 and coverage.get('defined_cases_pass') else 'PARTIAL'}] "
        f"{out_path} incomplete={incomplete}/{len(rows)} coverage={label}"
    )
    return 0 if incomplete == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
