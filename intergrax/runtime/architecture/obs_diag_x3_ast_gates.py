# © Artur Czarnecki. All rights reserved.

"""OBS-DIAG-X3 — AST gates for illegal diagnostic authority and tier-3 bypass patterns."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ObsDiagX3AstRuleId(StrEnum):
    LOCAL_DIAGNOSTIC_ORCHESTRATOR = "local_diagnostic_orchestrator"
    LOCAL_PROBLEM_LIFECYCLE = "local_problem_lifecycle"
    LOCAL_PROBLEM_GROUPING = "local_problem_grouping"
    LOCAL_EXECUTION_RECONSTRUCTOR = "local_execution_reconstructor"
    DIRECT_NEXUS_ROOT = "direct_nexus_root"
    DIRECT_PROBLEM_PERSISTENCE_MUTATION = "direct_problem_persistence_mutation"
    PRODUCT_OPERATOR_READ_NOT_ADOPTED = "product_operator_read_not_adopted"


@dataclass(frozen=True, slots=True)
class ObsDiagX3AstViolation:
    rule_id: ObsDiagX3AstRuleId
    relative_path: str
    line: int
    symbol: str
    message: str


_APPROVED_DIAGNOSTIC_COMPOSITION_RELATIVE = frozenset(
    {
        "intergrax/applications/_shared/diagnostic_composition.py",
        "intergrax/applications/_shared/diagnostic_runtime_wiring.py",
        "intergrax/applications/_shared/diagnostic_read_wiring.py",
        "intergrax/applications/_shared/harness_host_runtime.py",
        "intergrax/applications/_shared/product_observability_dashboard_wiring.py",
    }
)

_PRODUCT_OPERATOR_READ_WIRING_SYMBOL = "wire_harness_product_observability_dashboard"

_FORBIDDEN_LOCAL_AUTHORITY_SYMBOLS = frozenset(
    {
        "DiagnosticOrchestrator",
        "ProblemLifecycleEngine",
        "ProblemGroupingEngine",
        "ExecutionReconstructor",
    }
)

_DIRECT_PROBLEM_MUTATION_METHODS = frozenset(
    {
        "create",
        "create_problem",
        "upsert",
        "upsert_problem",
        "append_occurrence",
        "resolve_problem",
    }
)


def _relative_posix(path: Path, repo_root: Path) -> str:
    return path.relative_to(repo_root).as_posix()


def _is_approved_composition_file(relative: str) -> bool:
    return relative in _APPROVED_DIAGNOSTIC_COMPOSITION_RELATIVE


def _call_symbol(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _collect_call_violations_in_tree(
    *,
    tree: ast.AST,
    relative_path: str,
) -> list[ObsDiagX3AstViolation]:
    violations: list[ObsDiagX3AstViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        symbol = _call_symbol(node.func)
        if symbol is None:
            continue
        if symbol in _FORBIDDEN_LOCAL_AUTHORITY_SYMBOLS:
            rule = ObsDiagX3AstRuleId.LOCAL_DIAGNOSTIC_ORCHESTRATOR
            if symbol == "ProblemLifecycleEngine":
                rule = ObsDiagX3AstRuleId.LOCAL_PROBLEM_LIFECYCLE
            elif symbol == "ProblemGroupingEngine":
                rule = ObsDiagX3AstRuleId.LOCAL_PROBLEM_GROUPING
            elif symbol == "ExecutionReconstructor":
                rule = ObsDiagX3AstRuleId.LOCAL_EXECUTION_RECONSTRUCTOR
            violations.append(
                ObsDiagX3AstViolation(
                    rule_id=rule,
                    relative_path=relative_path,
                    line=node.lineno,
                    symbol=symbol,
                    message=f"{symbol}() is reserved for canonical composition modules",
                )
            )
        if symbol == "NexusLoop":
            violations.append(
                ObsDiagX3AstViolation(
                    rule_id=ObsDiagX3AstRuleId.DIRECT_NEXUS_ROOT,
                    relative_path=relative_path,
                    line=node.lineno,
                    symbol="NexusLoop",
                    message="direct NexusLoop() root construction bypasses harness host runtime",
                )
            )
        if isinstance(node.func, ast.Attribute) and node.func.attr in _DIRECT_PROBLEM_MUTATION_METHODS:
            parts: list[str] = []
            current: ast.expr = node.func.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            joined = ".".join(reversed(parts))
            if "ProblemPersistence" in joined or joined.endswith("_problem_persistence"):
                violations.append(
                    ObsDiagX3AstViolation(
                        rule_id=ObsDiagX3AstRuleId.DIRECT_PROBLEM_PERSISTENCE_MUTATION,
                        relative_path=relative_path,
                        line=node.lineno,
                        symbol=node.func.attr,
                        message=(
                            "application/scenario layers must not mutate ProblemPersistence "
                            "outside ProblemLifecycleEngine"
                        ),
                    )
                )
    return violations


def _scan_python_files(
    paths: list[Path],
    *,
    repo_root: Path,
) -> list[ObsDiagX3AstViolation]:
    violations: list[ObsDiagX3AstViolation] = []
    for path in paths:
        relative = _relative_posix(path, repo_root)
        if _is_approved_composition_file(relative):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        violations.extend(
            _collect_call_violations_in_tree(tree=tree, relative_path=relative),
        )
    return violations


def _iter_application_production_python(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    apps_root = repo_root / "applications"
    if not apps_root.is_dir():
        return paths
    for app_dir in sorted(apps_root.iterdir()):
        if not app_dir.is_dir() or app_dir.name.startswith("_"):
            continue
        for path in app_dir.rglob("*.py"):
            if "tests" in path.parts or path.name.startswith("test_"):
                continue
            rel = path.relative_to(repo_root).as_posix()
            if "/docker/" in rel or "runtime-context" in rel:
                continue
            paths.append(path)
    return paths


def _iter_scenario_application_python(repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    scenarios_root = repo_root / "platform_proofs" / "scenarios"
    if not scenarios_root.is_dir():
        return paths
    for scenario_dir in sorted(scenarios_root.iterdir()):
        app_dir = scenario_dir / "application"
        if not app_dir.is_dir():
            continue
        for path in app_dir.rglob("*.py"):
            paths.append(path)
    return paths


def _iter_worker_entry_python(repo_root: Path) -> list[Path]:
    from intergrax.applications._shared.execution_surface_discovery import (
        iter_worker_execution_surface_python_paths,
    )

    return list(iter_worker_execution_surface_python_paths(repo_root))


def collect_obs_diag_x3_production_layer_violations(
    repo_root: Path,
) -> list[ObsDiagX3AstViolation]:
    paths = (
        _iter_application_production_python(repo_root)
        + _iter_scenario_application_python(repo_root)
        + _iter_worker_entry_python(repo_root)
    )
    return _scan_python_files(paths, repo_root=repo_root)


def _factory_ast_calls_symbol(tree: ast.AST, symbol: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = _call_symbol(node.func)
        if called == symbol:
            return True
    return False


def collect_product_operator_read_adoption_violations(
    repo_root: Path,
) -> list[ObsDiagX3AstViolation]:
    """PRODUCT harness hosts must adopt canonical operator read via shared dashboard wiring."""
    from intergrax.applications._shared.application_runtime_graph import list_application_projects
    from intergrax.applications._shared.execution_surface_discovery import (
        resolve_application_profile,
    )
    from intergrax.applications.contracts.application_host import ApplicationProfile

    violations: list[ObsDiagX3AstViolation] = []
    apps_root = repo_root / "applications"
    if not apps_root.is_dir():
        return violations

    for app_name in list_application_projects(repo_root):
        profile = resolve_application_profile(repo_root, app_name)
        if profile is not ApplicationProfile.PRODUCT:
            continue
        factory_path = apps_root / app_name / "host" / "factory.py"
        if not factory_path.is_file():
            continue
        relative = _relative_posix(factory_path, repo_root)
        try:
            tree = ast.parse(factory_path.read_text(encoding="utf-8"), filename=str(factory_path))
        except SyntaxError:
            continue
        if _factory_ast_calls_symbol(tree, _PRODUCT_OPERATOR_READ_WIRING_SYMBOL):
            continue
        violations.append(
            ObsDiagX3AstViolation(
                rule_id=ObsDiagX3AstRuleId.PRODUCT_OPERATOR_READ_NOT_ADOPTED,
                relative_path=relative,
                line=1,
                symbol=_PRODUCT_OPERATOR_READ_WIRING_SYMBOL,
                message=(
                    "PRODUCT harness host factory must call "
                    "wire_harness_product_observability_dashboard"
                ),
            )
        )
    return violations


def collect_factory_entry_path_violations(repo_root: Path) -> list[str]:
    from scripts.gates.check_application_production_gates import (
        check_no_ad_hoc_nexus_in_factories,
    )

    violations = check_no_ad_hoc_nexus_in_factories(repo_root=repo_root)
    filtered: list[str] = []
    for item in violations:
        if (
            "must call build_harness_host_runtime" in item
            and "local_workspace_application" in item
        ):
            composition = (
                repo_root
                / "applications"
                / "local_workspace_application"
                / "host"
                / "host_runtime_composition.py"
            )
            if composition.is_file():
                text = composition.read_text(encoding="utf-8")
                if "build_harness_host_runtime" in text:
                    continue
        filtered.append(item)
    return filtered


__all__ = [
    "ObsDiagX3AstRuleId",
    "ObsDiagX3AstViolation",
    "collect_factory_entry_path_violations",
    "collect_obs_diag_x3_production_layer_violations",
    "collect_product_operator_read_adoption_violations",
]
