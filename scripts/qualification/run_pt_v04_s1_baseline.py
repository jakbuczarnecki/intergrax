#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
"""PT-V04 S1 diagnostic baseline runner (non-certifying; W2-RECOVERY-01)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_OUT = _REPO / ".tmp/session/w2-recovery-01/pt-v04"

PT_V04_TESTS = [
    "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_closure_counter_integrity.py",
    "tests/qualification/roadmap_replay_x/test_roadmap_replay_x_mechanical_exit_gates.py",
    "tests/qualification/state_x/test_state_x_p0_baseline.py",
    "tests/qualification/trace_x/test_trace_x_p0_baseline.py",
    "tests/qualification/trace_x/test_trace_x_p6_closed_world_gates.py",
    "tests/qualification/tenant_x/test_tenant_x_p0_fast.py",
]


def _sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO, text=True).strip()


def _run(cmd: list[str], label: str) -> dict[str, object]:
    proc = subprocess.run(cmd, cwd=_REPO, capture_output=True, text=True, check=False)
    return {
        "label": label,
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "stdout_tail": proc.stdout[-4000:] if proc.stdout else "",
        "stderr_tail": proc.stderr[-2000:] if proc.stderr else "",
    }


def main() -> int:
    _OUT.mkdir(parents=True, exist_ok=True)
    head = _sha()
    results: list[dict[str, object]] = []

    f10_refresh = _run(
        [sys.executable, "scripts/qualification/render_ebh7_f10_normative_matrix.py"],
        "F10_render",
    )
    results.append(f10_refresh)
    f10_check = _run(
        [sys.executable, "scripts/qualification/render_ebh7_f10_normative_matrix.py", "--check"],
        "F10_check",
    )
    results.append(f10_check)

    pytest_cmd = [sys.executable, "-m", "pytest", "-q", "--tb=line", *PT_V04_TESTS]
    results.append(_run(pytest_cmd, "PT_V04_pytest_suite"))

    traceability = _run([sys.executable, "scripts/governance/verify_arch_freeze_plan_traceability.py"], "traceability")
    results.append(traceability)

    summary = {
        "execution_sha": head,
        "recorded_at": datetime.now(UTC).isoformat(),
        "node": "PT-V04",
        "certifying": False,
        "results": results,
        "complete": all(r["exit_code"] == 0 for r in results),
    }
    out_json = _OUT / "PT_V04_S1_BASELINE.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"complete": summary["complete"], "path": str(out_json)}, indent=2))
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
