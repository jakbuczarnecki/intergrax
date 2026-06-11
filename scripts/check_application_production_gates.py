#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Tier-3 application production gate checks (APP-PROD-1..6)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"

REQUIRED_FACTORY_MARKERS = (
    "build_harness_host_runtime",
)

def _nexus_loop_call_lines(source: str) -> list[int]:
    """Return 1-based line numbers of direct ``NexusLoop(...)`` AST calls."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    hits: list[int] = []

    class _Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Name) and func.id == "NexusLoop":
                hits.append(node.lineno)
            elif isinstance(func, ast.Attribute) and func.attr == "NexusLoop":
                hits.append(node.lineno)
            self.generic_visit(node)

    _Visitor().visit(tree)
    return hits


def check_no_ad_hoc_nexus_in_factories() -> list[str]:
    violations: list[str] = []
    if not APPLICATIONS_ROOT.is_dir():
        return [f"missing {APPLICATIONS_ROOT}"]

    for path in APPLICATIONS_ROOT.glob("*_application/host/factory.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if not any(marker in text for marker in REQUIRED_FACTORY_MARKERS):
            violations.append(f"{rel}: must call build_harness_host_runtime")
        call_lines = _nexus_loop_call_lines(text)
        if call_lines:
            violations.append(
                f"{rel}: direct NexusLoop() at line(s) {call_lines} — use build_harness_host_runtime"
            )
    return violations


def check_manifest_profile_on_manifest() -> list[str]:
    violations: list[str] = []
    for path in APPLICATIONS_ROOT.glob("*_application/manifest.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if "ApplicationManifest" not in text:
            violations.append(f"{rel}: missing ApplicationManifest")
    return violations


def check_environment_wiring_entry() -> list[str]:
    violations: list[str] = []
    for path in APPLICATIONS_ROOT.glob("*_application/host/wiring.py"):
        rel = path.relative_to(REPO_ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        if "getattr(" in text and "manifest" in text:
            violations.append(f"{rel}: getattr on manifest is forbidden — use typed fields")
    return violations


def main() -> int:
    checks = (
        ("no_ad_hoc_nexus", check_no_ad_hoc_nexus_in_factories),
        ("manifest_profile_consistency", check_manifest_profile_on_manifest),
        ("environment_wiring", check_environment_wiring_entry),
    )
    violations: list[str] = []
    for _name, fn in checks:
        violations.extend(fn())

    if violations:
        print("application production gates: FAILED")
        for item in sorted(violations):
            print(f"  - {item}")
        return 1

    print("application production gates: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
