#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Tier-3 application production gate checks (APP-PROD-1..9 · APP-OPS-1..4 · APP-CON-7 · APP-EVOL-2..7)."""

from __future__ import annotations
from intergrax.utils import attribute_access

import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
APPLICATIONS_ROOT = REPO_ROOT / "applications"
for path in (REPO_ROOT, APPLICATIONS_ROOT, REPO_ROOT / "agents"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
_CI_DIR = REPO_ROOT / "scripts" / "ci"
if str(_CI_DIR) not in sys.path:
    sys.path.insert(0, str(_CI_DIR))
from script_paths import resolve_script  # noqa: E402

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
        if "attribute_access.optional(" in text and "manifest" in text:
            violations.append(f"{rel}: getattr on manifest is forbidden — use typed fields")
    return violations


def check_workspace_cleanup() -> list[str]:
    from intergrax.applications._shared.workspace_cleanup_wiring import (
        check_all_factory_workspace_cleanup,
    )

    return check_all_factory_workspace_cleanup(APPLICATIONS_ROOT)


def check_environment_state_usage() -> list[str]:
    from intergrax.applications._shared.environment_state_usage_wiring import (
        check_environment_state_usage as _check,
    )

    return _check(REPO_ROOT)


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


def check_application_recovery_contract() -> list[str]:
    from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
    from intergrax.applications._shared.recovery_contract_wiring import (
        check_strict_product_recovery_contract,
    )

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(
            check_strict_product_recovery_contract(
                product_id,
                manifest,
                applications_root=APPLICATIONS_ROOT,
            ),
        )
    return violations


def check_agent_certification_roster() -> list[str]:
    from intergrax.applications._shared.agent_certification_wiring import (
        check_strict_product_agent_certification,
    )
    from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        violations.extend(check_strict_product_agent_certification(product_id, manifest))
    return violations


def check_capability_alias_registry() -> list[str]:
    import importlib
    import inspect

    from intergrax.applications._shared.capability_alias_wiring import (
        check_environment_capability_aliases,
    )
    from intergrax.applications.contracts.manifest import ApplicationManifest

    violations: list[str] = []
    for package_dir in sorted(APPLICATIONS_ROOT.glob("*_application")):
        try:
            module = importlib.import_module(f"{package_dir.name}.manifest")
        except ImportError:
            continue
        manifest: ApplicationManifest | None = None
        for value in module.__dict__.values():
            if isinstance(value, ApplicationManifest):
                manifest = value
                break
        if manifest is None:
            for name in dir(module):
                if not (name.startswith("build_") and "manifest" in name.lower()):
                    continue
                builder = attribute_access.optional(module, name, None)
                if not callable(builder) or inspect.isclass(builder):
                    continue
                try:
                    signature = inspect.signature(builder)
                except (TypeError, ValueError):
                    continue
                required = [
                    param
                    for param in signature.parameters.values()
                    if param.default is inspect.Parameter.empty
                    and param.kind
                    in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                if required:
                    continue
                candidate = builder()
                if isinstance(candidate, ApplicationManifest):
                    manifest = candidate
                    break
        if manifest is None:
            continue
        env = manifest.resolved_environment()
        violations.extend(
            check_environment_capability_aliases(
                package_dir.name,
                manifest,
                env.capability_governance_profile,
            ),
        )
    return violations


def check_application_migrations() -> list[str]:
    from intergrax.applications._shared.migration_wiring import check_application_migrations as _check

    return _check(APPLICATIONS_ROOT)


def check_tier3_scenario_matrix() -> list[str]:
    from intergrax.applications._shared.tier3_scenario_matrix_wiring import (
        check_tier3_scenario_matrix as _check,
    )

    return _check(REPO_ROOT)


def check_tier3_audit_prompt() -> list[str]:
    import runpy

    gen = runpy.run_path(str(resolve_script("generate_domain_audit_prompts.py")))
    audit_path = REPO_ROOT / "docs" / "project" / "maintainers" / "audit" / "TIER3_APPLICATION_ENVIRONMENT.md"
    if not audit_path.is_file():
        return [f"missing audit prompt: {audit_path}"]
    tier3 = next(item for item in gen["DOMAINS"] if item["id"] == "TIER3_APPLICATION_ENVIRONMENT")
    expected = gen["render"](tier3)
    actual = audit_path.read_text(encoding="utf-8")
    if actual != expected:
        return ["tier3 audit prompt out of date — run generate_domain_audit_prompts.py"]
    return []


def check_application_registry() -> list[str]:
    from intergrax.applications._shared.registry_ops_wiring import check_platform_registries

    return check_platform_registries(REPO_ROOT)


def check_application_health_score() -> list[str]:
    from intergrax.applications._shared.health_score_wiring import check_strict_product_health_scores

    return check_strict_product_health_scores(REPO_ROOT)


def check_application_package() -> list[str]:
    from intergrax.applications._shared.environment_wiring import wire_application_environment
    from intergrax.applications._shared.package_wiring import (
        build_application_package,
        package_gate_environment,
        validate_application_package_closure,
    )
    from intergrax.applications._shared.product_manifest_registry import (
        iter_strict_product_manifests,
    )

    violations: list[str] = []
    for product_id, manifest in iter_strict_product_manifests():
        gate_env = package_gate_environment(manifest.resolved_environment())
        try:
            wiring = wire_application_environment(manifest, gate_env, conformance_check=False)
        except Exception as exc:
            violations.append(f"{product_id}: wire failed: {exc}")
            continue
        package = build_application_package(manifest, gate_env)
        violations.extend(
            validate_application_package_closure(
                package,
                manifest,
                gate_env,
                wiring.registry_snapshot,
                capability_graph=wiring.capability_graph,
            ),
        )
    return violations


def check_application_environment_diff() -> list[str]:
    from intergrax.applications._shared.environment_diff_wiring import (
        build_application_environment_diff,
    )
    from intergrax.applications._shared.product_manifest_registry import (
        iter_strict_product_manifests,
    )
    from intergrax.applications.contracts.application_environment_diff import DiffRiskLevel
    from intergrax.applications.contracts.execution_mode import ExecutionMode

    violations: list[str] = []
    manifests = list(iter_strict_product_manifests())
    if len(manifests) < 2:
        violations.append("need at least two STRICT product manifests for environment diff smoke")
        return violations

    for product_id, manifest in manifests:
        env = manifest.resolved_environment()
        self_diff = build_application_environment_diff(manifest, env, manifest, env)
        if self_diff.risk_level is not DiffRiskLevel.LOW:
            violations.append(f"{product_id}: self-diff risk {self_diff.risk_level.value}")
        if self_diff.breaking_changes:
            violations.append(f"{product_id}: self-diff must not report breaking changes")

    sample_manifest, sample_env = manifests[0][1], manifests[0][1].resolved_environment()
    mode_diff = build_application_environment_diff(
        sample_manifest,
        sample_env.model_copy(update={"execution_mode": ExecutionMode.BALANCED}),
        sample_manifest,
        sample_env.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
    )
    if mode_diff.risk_level is not DiffRiskLevel.HIGH:
        violations.append("execution_mode delta must classify as high risk")
    return violations


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
        ("environment_state_usage", check_environment_state_usage),
        ("workspace_cleanup", check_workspace_cleanup),
        ("capability_graph_strict_deploy", check_capability_graph_strict_deploy),
        ("application_ownership", check_application_ownership),
        ("tier3_scenario_matrix", check_tier3_scenario_matrix),
        ("application_migrations", check_application_migrations),
        ("capability_alias_registry", check_capability_alias_registry),
        ("agent_certification_roster", check_agent_certification_roster),
        ("application_recovery_contract", check_application_recovery_contract),
        ("application_environment_diff", check_application_environment_diff),
        ("application_package", check_application_package),
        ("application_health_score", check_application_health_score),
        ("application_registry", check_application_registry),
        ("tier3_audit_prompt", check_tier3_audit_prompt),
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
