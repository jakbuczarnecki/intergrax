"""VPI dependency-direction architecture conformance gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_VPI_ROOT = _REPO_ROOT / "platform_proofs/scenarios/verified_product_identification"


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_application_domain_does_not_import_integrations() -> None:
    domain_root = _VPI_ROOT / "application/domain"
    violations: list[str] = []
    for module_path in sorted(domain_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if imported.startswith("platform_proofs.scenarios.verified_product_identification.integrations"):
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []


def test_orchestration_depends_only_on_ports() -> None:
    orchestrator_path = _VPI_ROOT / "storage_bootstrap/orchestration/orchestrator.py"
    imports = _module_imports(orchestrator_path)
    forbidden = sorted(
        imported
        for imported in imports
        if imported.startswith(
            "platform_proofs.scenarios.verified_product_identification.integrations"
        )
    )
    assert forbidden == []


def test_storage_bootstrap_contracts_have_no_provider_paths() -> None:
    contracts_root = _VPI_ROOT / "storage_bootstrap/contracts"
    violations: list[str] = []
    for module_path in sorted(contracts_root.rglob("*.py")):
        for imported in _module_imports(module_path):
            if ".integrations." in imported:
                violations.append(f"{module_path.relative_to(_REPO_ROOT)} -> {imported}")
    assert violations == []
