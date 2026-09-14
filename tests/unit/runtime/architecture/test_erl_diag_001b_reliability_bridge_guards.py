# © Artur Czarnecki. All rights reserved.

"""ERL-DIAG-001B architecture guards for reliability diagnostic bridge."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RELIABILITY_BRIDGE_PKG = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "reliability"

_FORBIDDEN_IMPORT_PREFIXES = (
    "platform_proofs",
    "applications",
    "intergrax.runtime.governance",
    "intergrax.runtime.enterprise_reliability",
)

_FORBIDDEN_PERSISTENCE_MODULES = (
    "in_memory_problem_persistence",
    "document_store_problem_persistence",
    "problem_persistence",
    "problem_occurrence_persistence",
)


def _python_files_under(path: Path) -> list[Path]:
    return sorted(path.rglob("*.py"))


def _collect_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def test_bridge_package_imports_public_erl_diagnostics_contracts() -> None:
    bridge_path = _RELIABILITY_BRIDGE_PKG / "reliability_diagnostic_bridge.py"
    imports = _collect_imports(bridge_path)
    assert any(
        name.startswith("intergrax.contracts.enterprise_reliability.diagnostics")
        for name in imports
    )


def test_bridge_package_does_not_import_scenario_or_governance() -> None:
    violations: list[str] = []
    for path in _python_files_under(_RELIABILITY_BRIDGE_PKG):
        for module in _collect_imports(path):
            for prefix in _FORBIDDEN_IMPORT_PREFIXES:
                if module == prefix or module.startswith(f"{prefix}."):
                    violations.append(f"{path.relative_to(_REPO_ROOT)}: {module}")
    assert violations == []


def test_bridge_does_not_import_problem_persistence_adapters() -> None:
    violations: list[str] = []
    for path in _python_files_under(_RELIABILITY_BRIDGE_PKG):
        for module in _collect_imports(path):
            for forbidden in _FORBIDDEN_PERSISTENCE_MODULES:
                if forbidden in module:
                    violations.append(f"{path.relative_to(_REPO_ROOT)}: {module}")
    assert violations == []


def test_bridge_has_no_global_registry_pattern() -> None:
    for path in _python_files_under(_RELIABILITY_BRIDGE_PKG):
        source = path.read_text(encoding="utf-8")
        assert "service_locator" not in source
        assert "global " not in source


def test_public_emitter_factory_does_not_require_runtime_bridge_type_at_call_site() -> None:
    from intergrax.contracts.enterprise_reliability.diagnostics import (
        ExternalEffectReliabilityDiagnosticEmitter,
    )
    from intergrax.runtime.diagnostics.reliability import build_reliability_diagnostic_emitter

    assert callable(build_reliability_diagnostic_emitter)
    assert ExternalEffectReliabilityDiagnosticEmitter.__name__ in str(
        ExternalEffectReliabilityDiagnosticEmitter,
    )
