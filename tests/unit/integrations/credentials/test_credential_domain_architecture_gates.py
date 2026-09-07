# © Artur Czarnecki. All rights reserved.

"""Integration credential domain architecture gates (AW-7C P0-2)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_FORBIDDEN_IMPORT_PREFIXES = (
    "intergrax.autonomous_work",
    "intergrax.contracts.autonomous_work",
    "intergrax.nexus",
    "intergrax.runtime.nexus",
)


def _python_sources_under(package_name: str) -> list[Path]:
    package = importlib.import_module(package_name)
    assert package.__file__ is not None
    root = Path(package.__file__).parent
    return sorted(root.rglob("*.py"))


def _collect_imports(source: str) -> list[str]:
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def _assert_no_forbidden_imports(path: Path) -> None:
    joined = "\n".join(_collect_imports(path.read_text(encoding="utf-8"))).lower()
    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
        assert prefix.lower() not in joined, f"{path} imports forbidden surface {prefix}"


def test_credentials_package_forbidden_imports() -> None:
    for path in _python_sources_under("intergrax.integrations.credentials"):
        _assert_no_forbidden_imports(path)


def test_credential_contract_forbidden_imports() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    path = repo_root / "intergrax" / "integrations" / "contracts" / "credential.py"
    _assert_no_forbidden_imports(path)
