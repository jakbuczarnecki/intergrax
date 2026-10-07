# © Artur Czarnecki. All rights reserved.

"""AST/static helpers for TRACE-X-P5-R2-P1 qualification gates."""

from __future__ import annotations

import ast
from pathlib import Path

TRACE_X_P5_R2_P1_START_HEAD = "94099f9980a1bc15c95e779e426eb5a308bb290d"

_REPO_ROOT = Path(__file__).resolve().parents[3]

P1_CONTRACT_MODULES = (
    _REPO_ROOT / "intergrax/integrations/contracts/existing_capability_configuration_opportunity.py",
    _REPO_ROOT / "intergrax/integrations/contracts/execution_integration_configuration.py",
    _REPO_ROOT / "intergrax/contracts/execution_integration_configuration_provenance.py",
)

NEUTRAL_PROVENANCE_MODULE = (
    _REPO_ROOT / "intergrax/contracts/execution_integration_configuration_provenance.py"
)

FORBIDDEN_NEUTRAL_IMPORT_PREFIXES = (
    "intergrax.integrations.existing_capability_configuration_service",
    "intergrax.runtime.",
    "intergrax.applications.",
    "intergrax.autonomous_work.",
)

PRODUCTION_NO_MIGRATION_PATHS = (
    _REPO_ROOT / "intergrax/autonomous_work/worker_capability_fulfillment_coordinator.py",
    _REPO_ROOT / "intergrax/runtime/observability/reconstruction/execution_reconstructor.py",
)

P1_IMPORT_MARKERS = (
    "existing_capability_configuration_opportunity",
    "execution_integration_configuration",
    "execution_integration_configuration_provenance",
)


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_contains_any(path: Path, tokens: tuple[str, ...]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return [token for token in tokens if token in text]


def ast_names_defined_in_modules() -> set[str]:
    names: set[str] = set()
    for path in P1_CONTRACT_MODULES:
        tree = _read_ast(path)
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                names.add(node.name)
    return names


def forbidden_any_in_p1_modules() -> list[str]:
    violations: list[str] = []
    for path in P1_CONTRACT_MODULES:
        text = path.read_text(encoding="utf-8")
        if "Any" in text:
            tree = _read_ast(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == "Any":
                    violations.append(f"{path.name}: Any usage")
                if isinstance(node, ast.Attribute) and node.attr == "Any":
                    violations.append(f"{path.name}: Any attribute")
        if "dict[str, Any]" in text.replace(" ", ""):
            violations.append(f"{path.name}: dict[str, Any]")
        for probe in ("getattr(", "hasattr("):
            if probe in text:
                violations.append(f"{path.name}: {probe}")
    return violations


def neutral_provenance_import_violations() -> list[str]:
    tree = _read_ast(NEUTRAL_PROVENANCE_MODULE)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in FORBIDDEN_NEUTRAL_IMPORT_PREFIXES:
                if node.module.startswith(prefix):
                    violations.append(f"forbidden import: {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in FORBIDDEN_NEUTRAL_IMPORT_PREFIXES:
                    if alias.name.startswith(prefix):
                        violations.append(f"forbidden import: {alias.name}")
    return violations


def category_value_fallback_violations() -> list[str]:
    violations: list[str] = []
    for path in P1_CONTRACT_MODULES:
        text = path.read_text(encoding="utf-8")
        if "category.value" in text:
            violations.append(f"{path.name}: category.value fallback")
    return violations


def a0_low_mapping_violations() -> list[str]:
    violations: list[str] = []
    for path in P1_CONTRACT_MODULES:
        text = path.read_text(encoding="utf-8").lower()
        if "a0" in text and "low" in text:
            violations.append(f"{path.name}: suspected A0→LOW mapping")
    return violations


def reader_surface_violations() -> list[str]:
    path = NEUTRAL_PROVENANCE_MODULE
    tree = _read_ast(path)
    violations: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "ExecutionIntegrationConfigurationProvenanceReader":
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                if item.name in {"read_latest", "write", "pin"}:
                    violations.append(f"forbidden reader method: {item.name}")
    text = path.read_text(encoding="utf-8")
    if "read_latest" in text:
        violations.append("read_latest present in neutral provenance module")
    return violations


def configured_binding_duplication_violations() -> list[str]:
    path = _REPO_ROOT / "intergrax/integrations/contracts/execution_integration_configuration.py"
    tree = _read_ast(path)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "ConfiguredCapabilityBinding":
            return ["ConfiguredCapabilityBinding duplicated in execution_integration_configuration"]
    return []


def production_caller_migration_violations() -> list[str]:
    violations: list[str] = []
    for path in PRODUCTION_NO_MIGRATION_PATHS:
        if not path.is_file():
            continue
        hits = module_contains_any(path, P1_IMPORT_MARKERS)
        if hits:
            violations.append(f"{path.relative_to(_REPO_ROOT)}: imports {hits}")
    return violations
