#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""MVP promotion gates G0–G2 (MVP-EVOL.1)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _exists(relative: str) -> bool:
    return (REPO_ROOT / relative).is_file()


def gate_g0_runnable() -> tuple[bool, str]:
    required = [
        "intergrax/scaffold/cli.py",
        "intergrax/cli/doctor.py",
        "intergrax/cli/run.py",
        "applications/lab_application/host/factory.py",
    ]
    missing = [path for path in required if not _exists(path)]
    if missing:
        return False, f"missing runnable artifacts: {', '.join(missing)}"
    return True, "scaffold + lab host present"


def gate_g1_eval_baseline() -> tuple[bool, str]:
    eval_modules = [
        "intergrax/runtime/architecture/evaluation_automation.py",
        "intergrax/runtime/architecture/online_evaluation_registry.py",
        "scripts/check_harness_evaluation_wiring.py",
    ]
    missing = [path for path in eval_modules if not _exists(path)]
    if missing:
        return False, f"missing evaluation baseline: {', '.join(missing)}"
    return True, "evaluation control plane present"


def gate_g2_policy() -> tuple[bool, str]:
    policy_modules = [
        "intergrax/contracts/resilience_policy.py",
        "intergrax/contracts/autonomy_level.py",
        "intergrax/runtime/policy/autonomy_resolver.py",
        "scripts/check_harness_resilience_policy.py",
    ]
    missing = [path for path in policy_modules if not _exists(path)]
    if missing:
        return False, f"missing policy modules: {', '.join(missing)}"
    return True, "resilience + autonomy policy present"


def main() -> int:
    gates = [
        ("G0", gate_g0_runnable),
        ("G1", gate_g1_eval_baseline),
        ("G2", gate_g2_policy),
    ]
    failures: list[str] = []
    for gate_id, gate_fn in gates:
        ok, detail = gate_fn()
        status = "OK" if ok else "FAIL"
        print(f"{gate_id}: {status} — {detail}")
        if not ok:
            failures.append(gate_id)

    if failures:
        return 1

    if "--with-doctor" in sys.argv[1:]:
        doctor = subprocess.run(
            [sys.executable, "-m", "intergrax.cli.doctor"],
            cwd=REPO_ROOT,
            check=False,
        )
        if doctor.returncode != 0:
            print("doctor smoke failed")
            return doctor.returncode

    print("mvp promotion gates G0–G2: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
