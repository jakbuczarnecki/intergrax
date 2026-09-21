# © Artur Czarnecki. All rights reserved.

"""UCA-6A — architecture gates for CodeCraft UCA acquisition adapter."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STRATEGY_PATH = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "codecraft"
    / "acquisition"
    / "uca_acquisition_strategy.py"
)
_COORDINATION_ROOT = _REPO_ROOT / "intergrax" / "capability_acquisition"

_FORBIDDEN_PREFIXES = (
    "intergrax.autonomous_work",
    "intergrax.marketplace",
    "intergrax.runtime.execution",
    "intergrax.tools.registry",
    "intergrax.skills.registry",
    "intergrax.agent_distribution",
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


def test_codecraft_strategy_avoids_forbidden_dependencies() -> None:
    imports = _module_imports(_STRATEGY_PATH)
    for module in imports:
        for forbidden in _FORBIDDEN_PREFIXES:
            assert not module.startswith(forbidden), (
                f"codecraft uca strategy imports forbidden {module}"
            )


def test_uca_coordination_does_not_import_codecraft_implementation() -> None:
    paths = [p for p in _COORDINATION_ROOT.rglob("*.py") if p.name != "__init__.py"]
    for path in paths:
        for module in _module_imports(path):
            assert not module.startswith("intergrax.runtime.codecraft"), (
                f"{path.relative_to(_REPO_ROOT)} imports {module}"
            )
            assert not module.startswith("intergrax.contracts.codecraft"), (
                f"{path.relative_to(_REPO_ROOT)} imports {module}"
            )
