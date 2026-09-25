# © Artur Czarnecki. All rights reserved.

"""UCA-6C-R6-CERT — ToolRegistry runtime leaf import must not load composition graph."""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_REGISTRY_INIT = _REPO_ROOT / "intergrax" / "tools" / "registry" / "__init__.py"
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_FORBIDDEN_PREFIXES = (
    "intergrax.tools.registry.wiring",
    "intergrax.tools.registry.catalog",
    "intergrax.tools.registry.bootstrap",
    "intergrax.tools.registry.factory",
    "intergrax.integrations.registry",
    "intergrax.integrations.providers.relational_store.sqlite",
    "intergrax.collaborative_work",
    "intergrax.runtime.integrations.categories",
)

_LIGHTWEIGHT_EXPORTS = frozenset(
    {
        "RegisteredTool",
        "ToolProfile",
        "ToolRegistry",
        "ToolRegistryRead",
        "default_lab_tool_profile",
    }
)

_COMPOSITION_ROOT_SYMBOLS = frozenset(
    {
        "ToolWiringContext",
        "build_registry_from_profile",
        "enabled_tool_ids_for_profile",
        "register_default_tools",
        "reset_default_tools_bootstrap",
        "bootstrap_catalogs",
        "register_tool_catalog",
        "register_tool_plugin",
    }
)


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_tool_registry_runtime_cold_import_in_subprocess() -> None:
    forbidden_literal = repr(_FORBIDDEN_PREFIXES)
    statement = (
        "import sys\n"
        "from intergrax.tools.registry.runtime import ToolRegistry\n"
        f"forbidden = {forbidden_literal}\n"
        "loaded = [name for name in sys.modules if any(name.startswith(p) for p in forbidden)]\n"
        "assert not loaded, f'unexpected modules loaded: {loaded}'\n"
        "assert ToolRegistry is not None\n"
    )
    completed = _run_import_subprocess(statement)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_tools_registry_root_all_is_lightweight() -> None:
    package = importlib.import_module("intergrax.tools.registry")
    assert frozenset(package.__all__) == _LIGHTWEIGHT_EXPORTS


def test_tools_registry_init_does_not_reexport_composition() -> None:
    tree = ast.parse(
        _REGISTRY_INIT.read_text(encoding="utf-8"),
        filename=str(_REGISTRY_INIT),
    )
    imported: set[str] = set()
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module is None:
            continue
        if not node.module.startswith("intergrax.tools.registry."):
            continue
        leaf = node.module.removeprefix("intergrax.tools.registry.")
        if leaf in {"wiring", "factory", "bootstrap", "catalog"}:
            for alias in node.names:
                imported.add(alias.name)
    assert not imported, (
        f"root registry re-exports composition symbols: {sorted(imported)}"
    )


def test_tools_registry_init_has_no_dynamic_export_routing() -> None:
    text = _REGISTRY_INIT.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(_REGISTRY_INIT))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert node.name != "__getattr__", (
                "tools.registry must not define __getattr__"
            )
    assert "importlib" not in text
    assert "lazy_export" not in text


def test_production_code_does_not_import_composition_from_registry_root() -> None:
    violations: list[str] = []
    for path in sorted(_INTERGRAX_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "intergrax.tools.registry":
                continue
            for alias in node.names:
                name = alias.name
                if name in _COMPOSITION_ROOT_SYMBOLS:
                    rel = path.relative_to(_REPO_ROOT)
                    violations.append(
                        f"{rel}:{node.lineno} imports {name} from registry root"
                    )
    assert not violations, "\n".join(violations)
