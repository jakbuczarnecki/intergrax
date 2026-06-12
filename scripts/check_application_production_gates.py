#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Tier-3 application production gate checks (APP-PROD-1..8 · APP-OPS-1..2 · APP-CON-7)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

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


def check_workspace_cleanup() -> list[str]:
    from intergrax.applications._shared.workspace_cleanup_wiring import (
        check_all_factory_workspace_cleanup,
    )

    return check_all_factory_workspace_cleanup(APPLICATIONS_ROOT)


def check_application_ownership() -> list[str]:
    from intergrax.applications._shared.ownership_wiring import (
        check_manifest_operational_ownership,
    )
    from intergrax.applications._shared.product_manifest_registry import (
        iter_product_manifests,
    )

    violations: list[str] = []
    for product_id, manifest in iter_product_manifests():
        violations.extend(check_manifest_operational_ownership(product_id, manifest))
    return violations


def check_capability_graph_strict_deploy() -> list[str]:
    from intergrax.applications._shared.capability_graph_deploy_gate import (
        check_strict_product_capability_graph,
    )
    from intergrax.applications._shared.product_manifest_registry import (
        iter_strict_product_manifests,
    )

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(check_strict_product_capability_graph(product_id, manifest))
    return violations


def check_tier3_scenario_matrix() -> list[str]:
    from intergrax.applications._shared.tier3_scenario_matrix_wiring import (
        check_tier3_scenario_matrix as _check,
    )

    return _check(REPO_ROOT)


def check_budget_enforcement() -> list[str]:
    from intergrax.applications._shared.budget_wiring import check_manifest_budget_enforcement
    from intergrax.applications._shared.product_manifest_registry import (
        iter_strict_product_manifests,
    )

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(check_manifest_budget_enforcement(product_id, manifest))
    return violations


def main() -> int:
    checks = (
        ("no_ad_hoc_nexus", check_no_ad_hoc_nexus_in_factories),
        ("manifest_profile_consistency", check_manifest_profile_on_manifest),
        ("environment_wiring", check_environment_wiring_entry),
        ("budget_enforcement", check_budget_enforcement),
        ("workspace_cleanup", check_workspace_cleanup),
        ("capability_graph_strict_deploy", check_capability_graph_strict_deploy),
        ("application_ownership", check_application_ownership),
        ("tier3_scenario_matrix", check_tier3_scenario_matrix),
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
