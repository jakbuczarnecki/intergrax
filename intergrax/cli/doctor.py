# © Artur Czarnecki. All rights reserved.

"""``intergrax doctor`` — harness health checks (Phase DX-3.3, DX-8.1)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import argparse
import subprocess
import sys
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("doctor", help="Harness health checks and Tier-3 environment diff")
    parser.add_argument("--ci", action="store_true", help="Exit non-zero on violations")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    doctor_sub = parser.add_subparsers(dest="doctor_command")
    doctor_sub.add_parser("check", help="Run harness health scripts (default)")
    from intergrax.cli.doctor_diff_app import register_parser as register_diff_app
    from intergrax.cli.doctor_health_app import register_parser as register_health_app

    register_diff_app(doctor_sub)
    register_health_app(doctor_sub)


def _run_script(script: Path, root: Path) -> tuple[bool, str]:
    if not script.is_file():
        return True, f"skip missing {script.name}"
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip() or "ok"
    return False, proc.stderr.strip() or proc.stdout.strip() or f"exit {proc.returncode}"


def run_doctor(args: argparse.Namespace) -> int:
    if attribute_access.optional(args, "doctor_command", None) == "diff-app":
        from intergrax.cli.doctor_diff_app import run_doctor_diff_app

        return run_doctor_diff_app(args)
    if attribute_access.optional(args, "doctor_command", None) == "health-app":
        from intergrax.cli.doctor_health_app import run_doctor_health_app

        return run_doctor_health_app(args)

    root = args.root.resolve()
    checks: list[tuple[str, bool, str]] = []

    scripts = [
        ("agent_registry_bypass", "check_agent_registry_bypass.py"),
        ("agents_no_tier3_imports", "check_agents_no_tier3_imports.py"),
        ("scaffold_harness_alignment", "check_scaffold_harness_alignment.py"),
        ("harness_no_getattr", "check_harness_no_getattr.py"),
        ("harness_guardrail_wiring", "check_harness_guardrail_wiring.py"),
        ("context_engine_wiring", "check_context_engine_wiring.py"),
        ("llm_adapter_typed_returns", "check_llm_adapter_typed_returns.py"),
        ("llm_profile_runtime", "check_llm_profile_runtime.py"),
        ("model_catalog_coverage", "check_model_catalog_coverage.py"),
        ("tool_injection_defense", "check_tool_injection_defense.py"),
        ("legacy_tool_plan_booleans", "check_legacy_tool_plan_booleans.py"),
        ("runtime_event_tenant", "check_runtime_event_tenant_propagation.py"),
        ("audit_ideal_dx_subset", "gates/check_audit_ideal_gates.py"),
        ("skill_selection_hook", "check_skill_selection_hook.py"),
    ]
    for name, script_name in scripts:
        ok, msg = _run_script(root / "scripts" / script_name, root)
        checks.append((name, ok, msg))

    maturity_script = root / "scripts" / "harness_maturity_report.py"
    if maturity_script.is_file():
        proc = subprocess.run(
            [sys.executable, str(maturity_script), "--json"],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            import json

            report = json.loads(proc.stdout)
            l3 = report.get("l3_layers", 0)
            total = report.get("total_layers", 32)
            checks.append(("harness_maturity_score", True, f"{l3}/{total} layers L3+"))
        else:
            checks.append(("harness_maturity_score", False, proc.stderr.strip() or "failed"))

    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}: {detail}")

    failed = [c for c in checks if not c[1]]
    if failed and args.ci:
        return 1
    if failed:
        print(f"\n{len(failed)} check(s) failed (run with --ci to enforce in CI).")
    else:
        print("\nAll doctor checks passed.")
    return 0
