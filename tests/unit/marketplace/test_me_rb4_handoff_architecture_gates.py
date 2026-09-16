# © Artur Czarnecki. All rights reserved.

"""ME-RB4 — lifecycle handoff architecture import gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_HANDOFF_CORE_PREFIX = "intergrax.marketplace.handoff"
_ADAPTER_PREFIX = "intergrax.marketplace.handoff.adapters"

_FORBIDDEN_CORE_PREFIXES = (
    "intergrax.agent_distribution",
    "intergrax.tools.registry.runtime",
    "intergrax.skills.registry.runtime",
    "intergrax.runtime.execution",
    "intergrax.runtime.nexus",
    "intergrax.nexus",
)


def _package_root(module_name: str) -> Path:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return Path(package.__path__[0])


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _collect_imports(tree: ast.AST) -> list[str]:
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_handoff_core_does_not_import_domain_implementations() -> None:
    root = _package_root("intergrax.marketplace.handoff")
    adapter_root = root / "adapters"
    for path in _iter_py_files(root):
        if adapter_root in path.parents or path.parent == adapter_root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in _FORBIDDEN_CORE_PREFIXES:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"handoff core {path.relative_to(root)} imports forbidden: {imported}",
                    )


def test_handoff_adapters_do_not_import_runtime_registries() -> None:
    root = _package_root("intergrax.marketplace.handoff.adapters")
    forbidden = (
        "intergrax.tools.registry.runtime",
        "intergrax.skills.registry.runtime",
        "intergrax.runtime.nexus",
        "intergrax.nexus",
    )
    for path in _iter_py_files(root):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _collect_imports(tree):
            for prefix in forbidden:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(
                        f"adapter {path.relative_to(root)} imports forbidden: {imported}",
                    )


def test_agent_bridge_imports_agent_distribution_acquisition_contract_only() -> None:
    bridge = _package_root("intergrax.marketplace.handoff.adapters") / "agent_distribution_bridge.py"
    tree = ast.parse(bridge.read_text(encoding="utf-8"))
    agent_distribution_imports = [
        imported
        for imported in _collect_imports(tree)
        if imported == "intergrax.agent_distribution"
        or imported.startswith("intergrax.agent_distribution.")
    ]
    assert agent_distribution_imports == ["intergrax.agent_distribution.dynamic_acquisition"]
