#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""AUDIT-IDEAL-26.1 — umbrella gate for post-L3 ideal architecture closeout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(script: str, *extra: str) -> int:
    script_path = str(REPO_ROOT / "scripts" / script)
    for cmd in (
        ["uv", "run", "python", script_path, *extra],
        [PYTHON, script_path, *extra],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            return 0
    print(f"FAILED: {script}", file=sys.stderr)
    return 1


def main() -> int:
    scripts = [
        ("harness_maturity_report.py", ("--enforce-l3-critical",)),
        ("check_shadow_eval_automation.py", ()),
        ("check_product_release_eval_gate.py", ()),
        ("check_registry_snapshot_diff.py", ()),
        ("check_context_golden.py", ()),
        ("check_agents_lifecycle_metadata.py", ()),
        ("check_structured_output_gate.py", ()),
        ("check_product_security_wiring.py", ()),
        ("check_sandbox_policy_wiring.py", ()),
        ("check_agent_promotion_eval_gate.py", ()),
        ("check_l4_runtime_evidence.py", ()),
        ("check_tenant_storage_isolation.py", ()),
        ("check_context_drift_monitoring.py", ()),
        ("check_modality_live_endpoints.py", ()),
        ("check_deploy_slo_evidence.py", ()),
        ("check_product_intake_parity.py", ()),
        ("check_pre_context_policy_wiring.py", ()),
        ("check_tool_injection_defense.py", ()),
        ("phase_v_capability_graph_guard.py", ("--enforce",)),
    ]
    exit_code = 0
    for script, extra in scripts:
        exit_code = exit_code or _run(script, *extra)
    test_path = REPO_ROOT / "tests" / "unit" / "runtime" / "architecture" / "test_audit_ideal_depth_gate.py"
    for cmd in (
        ["uv", "run", "pytest", str(test_path), "-q", "-m", "gate"],
        [PYTHON, "-m", "pytest", str(test_path), "-q", "-m", "gate"],
    ):
        completed = subprocess.run(cmd, cwd=REPO_ROOT, check=False)
        if completed.returncode == 0:
            break
    else:
        print("FAILED: test_audit_ideal_depth_gate.py", file=sys.stderr)
        exit_code = 1
    if exit_code == 0:
        print("OK: AUDIT-IDEAL gate checks passed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
