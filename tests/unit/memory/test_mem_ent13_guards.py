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


def test_qualification_package_has_no_rag_imports() -> None:
    for path in _iter_py_files(_QUAL_ROOT):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("intergrax.rag"), f"{path} imports intergrax.rag"


_SESSION_TURN_INDEX_CONTRACT = (
    _REPO / "intergrax" / "memory" / "contracts" / "session_turn_index.py"
)


def _function_kwonly_uses_any(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            for arg in node.args.kwonlyargs:
                if arg.arg == "kwargs" and arg.annotation is not None:
                    if _annotation_uses_name(arg.annotation, "Any"):
                        return True
            if node.args.vararg and node.args.vararg.arg == "kwargs":
                return True
    return False


def _annotation_uses_name(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name) and node.id == name:
        return True
    if isinstance(node, ast.Subscript):
        return _annotation_uses_name(node.value, name) or _annotation_uses_name(
            node.slice, name
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_uses_name(node.left, name) or _annotation_uses_name(node.right, name)
    if isinstance(node, ast.Tuple):
        return any(_annotation_uses_name(elt, name) for elt in node.elts)
    return False


def _collect_public_contract_annotation_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            for label in ("object", "Any"):
                if _annotation_uses_name(node.annotation, label):
                    names.add(label)
        if isinstance(node, ast.AnnAssign) and node.annotation is not None:
            for label in ("object", "Any"):
                if _annotation_uses_name(node.annotation, label):
                    names.add(label)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.returns is not None:
            if _annotation_uses_name(node.returns, "Any"):
                names.add("Any")
            if _annotation_uses_name(node.returns, "dict"):
                if isinstance(node.returns, ast.Subscript):
                    names.add("raw_dict_result")
    return names


def test_session_turn_index_public_contract_is_fully_typed() -> None:
    path = _SESSION_TURN_INDEX_CONTRACT
    names = _collect_public_contract_annotation_names(path)
    assert "Any" not in names
    assert "object" not in names
    assert "raw_dict_result" not in names
    assert not _function_kwonly_uses_any(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "search_turns":
            assert node.returns is not None
            assert _annotation_uses_name(node.returns, "SessionTurnIndexHit")


def test_canonical_check_suites_have_unique_required_ids() -> None:
    from intergrax.memory.provider_qualification.checks import (
        ENTITY_TEMPORAL_MEMORY_STORE_CHECKS,
        LONG_HORIZON_MEMORY_STORE_CHECKS,
        PROCEDURE_MEMORY_STORE_CHECKS,
        SESSION_TURN_INDEX_STORE_CHECKS,
        USER_PROFILE_STORE_CHECKS,
    )

    for suite in (
        USER_PROFILE_STORE_CHECKS,
        ENTITY_TEMPORAL_MEMORY_STORE_CHECKS,
        PROCEDURE_MEMORY_STORE_CHECKS,
        LONG_HORIZON_MEMORY_STORE_CHECKS,
        SESSION_TURN_INDEX_STORE_CHECKS,
    ):
        ids = [item.check_id for item in suite]
        assert len(ids) == len(set(ids))
        assert all(item.strip() for item in ids)
