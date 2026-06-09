# © Artur Czarnecki. All rights reserved.

"""``intergrax doctor`` — harness health checks (Phase DX-3.3, DX-8.1)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser("doctor", help="Check tier imports, scaffold alignment, optional CI mode")
    parser.add_argument("--ci", action="store_true", help="Exit non-zero on violations")
    parser.add_argument("--root", type=Path, default=Path.cwd())


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
    root = args.root.resolve()
    checks: list[tuple[str, bool, str]] = []

    scripts = [
        ("agent_registry_bypass", "check_agent_registry_bypass.py"),
        ("agents_no_tier3_imports", "check_agents_no_tier3_imports.py"),
        ("scaffold_harness_alignment", "check_scaffold_harness_alignment.py"),
        ("harness_no_getattr", "check_harness_no_getattr.py"),
        ("harness_guardrail_wiring", "check_harness_guardrail_wiring.py"),
    ]
    for name, script_name in scripts:
        ok, msg = _run_script(root / "scripts" / script_name, root)
        checks.append((name, ok, msg))

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
