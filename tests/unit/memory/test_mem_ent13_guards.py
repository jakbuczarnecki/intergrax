# © Artur Czarnecki. All rights reserved.

"""MEM-ENT-13 architecture guards."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_QUAL_ROOT = _REPO / "intergrax" / "memory" / "provider_qualification"
_CONTRACT = _REPO / "intergrax" / "memory" / "contracts" / "provider_qualification.py"

_FORBIDDEN_IMPORTS = (
    "sqlite3",
    "mongodb",
    "redis",
    "qdrant",
    "pinecone",
)

_FORBIDDEN_CALLS = ("getattr", "setattr", "getmembers")


def _iter_py_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _call_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            names.add(node.func.id)
    return names


def test_qualification_core_has_no_vendor_imports() -> None:
    files = _iter_py_files(_QUAL_ROOT) + [_CONTRACT]
    for path in files:
        imported = _module_imports(path)
        for forbidden in _FORBIDDEN_IMPORTS:
            assert forbidden not in imported, f"{path} imports {forbidden}"


def test_qualification_core_has_no_reflection_calls() -> None:
    files = _iter_py_files(_QUAL_ROOT) + [_CONTRACT]
    for path in files:
        calls = _call_names(path)
        for forbidden in _FORBIDDEN_CALLS:
            assert forbidden not in calls, f"{path} uses {forbidden}()"


def test_public_qualification_contract_has_no_any_annotation() -> None:
    text = _CONTRACT.read_text(encoding="utf-8")
    assert "Any" not in text
    assert "dict[str," not in text


def test_qualification_result_has_no_score_authority_field() -> None:
    tree = ast.parse(_CONTRACT.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assert node.target.id != "score"


def _imports_intergrax_runtime(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("intergrax.runtime"):
                return True
    return False


def test_qualification_package_has_no_runtime_imports() -> None:
    for path in _iter_py_files(_QUAL_ROOT):
        assert not _imports_intergrax_runtime(path), f"{path} imports intergrax.runtime"
