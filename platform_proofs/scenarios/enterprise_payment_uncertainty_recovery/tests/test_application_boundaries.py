"""Dependency boundary checks for the scenario application package."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCENARIO_ROOT = Path(__file__).resolve().parents[1]
_APPLICATION_ROOT = _SCENARIO_ROOT / "application"

_FORBIDDEN_ROOTS = frozenset(
    {
        "psycopg",
        "asyncpg",
        "sqlalchemy",
        "platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.provisioning",
        "platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.database",
    }
)


def _imported_module_roots(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def test_application_package_does_not_import_storage_adapters() -> None:
    violations: list[str] = []
    for path in _APPLICATION_ROOT.rglob("*.py"):
        for module in _imported_module_roots(path):
            lowered = module.casefold()
            for forbidden in _FORBIDDEN_ROOTS:
                if lowered == forbidden.casefold() or lowered.startswith(forbidden.casefold() + "."):
                    violations.append(f"{path.relative_to(_SCENARIO_ROOT)} imports {module}")
    assert not violations, "\n".join(violations)
