# © Artur Czarnecki. All rights reserved.

"""PLUG-02 — public extension boundary static gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]

_TOOLS_PUBLIC_CONTRACT_MODULES = (
    "intergrax/tools/invocation_wiring.py",
    "intergrax/tools/invocation_wiring_adapter.py",
    "intergrax/tools/invocation_wiring_requirements.py",
    "intergrax/tools/registry/session_storage_binding.py",
    "intergrax/tools/invocation_pattern/contracts.py",
    "intergrax/tools/invocation_pattern/registry.py",
)

_MEMORY_CONTRACTS_DIR = _REPO_ROOT / "intergrax" / "memory" / "contracts"

_REFERENCE_PLUGIN_ROOT = (
    _REPO_ROOT
    / "examples"
    / "platform_plugins"
    / "intergrax_reference_enterprise_plugin"
    / "src"
)

pytestmark = pytest.mark.unit


def _module_imports_nexus(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime.nexus"):
                hits.append(node.module)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("intergrax.runtime.nexus"):
                    hits.append(alias.name)
    return hits


@pytest.mark.parametrize("relative", _TOOLS_PUBLIC_CONTRACT_MODULES)
def test_tools_public_contract_modules_do_not_import_nexus(relative: str) -> None:
    path = _REPO_ROOT / Path(relative)
    hits = _module_imports_nexus(path)
    assert hits == [], f"{relative} must not import Nexus: {hits}"


def test_memory_contracts_do_not_reference_nexus_types() -> None:
    violations: list[str] = []
    for path in _MEMORY_CONTRACTS_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "intergrax.runtime.nexus" in text:
            violations.append(path.name)
    assert violations == []


def test_reference_enterprise_plugin_does_not_import_nexus() -> None:
    violations: list[str] = []
    for path in _REFERENCE_PLUGIN_ROOT.rglob("*.py"):
        hits = _module_imports_nexus(path)
        if hits:
            violations.append(f"{path.relative_to(_REPO_ROOT)}: {hits}")
    assert violations == []
