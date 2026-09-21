# © Artur Czarnecki. All rights reserved.

"""UCA-4 — import architecture gates for qualification coordination."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_COORDINATION_ROOT = _REPO_ROOT / "intergrax" / "capability_qualification"
_CONTRACTS_ROOT = _REPO_ROOT / "intergrax" / "contracts" / "capability_qualification"

_FORBIDDEN_PREFIXES = (
    "intergrax.marketplace",
    "intergrax.runtime.codecraft",
    "intergrax.runtime.execution",
    "intergrax.tools.registry.runtime",
    "intergrax.skills.registry",
    "intergrax.agent_distribution.admin_service",
    "intergrax.capability_acquisition.availability_projection",
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_qualification_coordination_avoids_forbidden_dependencies() -> None:
    paths = [
        p
        for p in list(_COORDINATION_ROOT.rglob("*.py"))
        + list(_CONTRACTS_ROOT.rglob("*.py"))
        if p.name != "__init__.py"
    ]
    assert paths
    for path in paths:
        for module in _module_imports(path):
            for forbidden in _FORBIDDEN_PREFIXES:
                assert not module.startswith(forbidden), (
                    f"{path.relative_to(_REPO_ROOT)} imports forbidden {module}"
                )


def test_qualification_contracts_do_not_reference_host_available() -> None:
    for path in _CONTRACTS_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "HOST_AVAILABLE" not in text, path.name
