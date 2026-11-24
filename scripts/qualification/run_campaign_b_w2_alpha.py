#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""Campaign B-1/B-2 alpha runner (diagnostic; W2-RECOVERY-01)."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_MANIFEST = Path(__file__).with_name("campaign_b_w2_manifest.v1.json")
_OUT = _REPO / ".tmp/session/w2-recovery-01/campaign-b"


def _sha(cwd: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd, text=True).strip()


def _run_case(case: dict[str, object], cwd: Path) -> dict[str, object]:
    cmd_s = str(case["command"]).replace("python ", f"{sys.executable} ", 1)
    cmd = shlex.split(cmd_s, posix=False)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "case_id": case["case_id"],
        "contract": case.get("contract"),
        "command": cmd_s,
        "sha": _sha(cwd),
        "result": "PASS" if proc.returncode == 0 else "FAIL",
        "exit_code": proc.returncode,
        "evidence_dir": str(_OUT),
        "finding": None if proc.returncode == 0 else (proc.stderr or proc.stdout)[-500:],
        "owner": "PT-B01" if case["case_id"].startswith("R0") or case["case_id"] in {"R17"} else "PT-B02",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("b1", "b2"), required=True)
    parser.add_argument("--worktree", type=Path, default=None, help="detached worktree @ B for b1")
    args = parser.parse_args(argv)

    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    cases = manifest["b1_cases"] if args.phase == "b1" else manifest["b2_cases"]
    _OUT.mkdir(parents=True, exist_ok=True)

    cwd = _REPO
    if args.phase == "b1" and args.worktree and args.worktree.is_dir():
        cwd = args.worktree

    rows = [_run_case(c, cwd) for c in cases]
    out = {
        "phase": args.phase,
        "recorded_at": datetime.now(UTC).isoformat(),
        "cwd": str(cwd),
        "b1_baseline_sha": manifest.get("b1_baseline_sha"),
        "rows": rows,
    }
    out_path = _OUT / f"RESULTS_{args.phase.upper()}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for row in rows:
        print(
            f"{row['case_id']} | {row['contract']} | exit={row['exit_code']} | SHA={row['sha'][:12]} | {row['result']}"
        )
    fails = sum(1 for r in rows if r["exit_code"] != 0)
    print(f"[{'OK' if fails == 0 else 'PARTIAL'}] {out_path} fails={fails}/{len(rows)}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
