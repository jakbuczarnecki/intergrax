# © Artur Czarnecki. All rights reserved.

"""AST type-quality gate for vector index administration control-plane path."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_SCOPED_PRODUCTION_FILES = (
    "intergrax/integrations/contracts/vector_index_administration.py",
    "intergrax/integrations/providers/vector_store/qdrant/index_administration.py",
)

_FORBIDDEN_CALLS = frozenset({"getattr", "setattr", "hasattr"})
_FORBIDDEN_IMPORTS = frozenset({"inspect"})
_FORBIDDEN_TYPING_NAMES = frozenset({"Any"})
_FORBIDDEN_SUBSCRIPT_KEYS = frozenset(
    {
        ("Mapping", "object"),
        ("dict", "Any"),
        ("dict", "object"),
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _annotation_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        if base is None:
            return None
        if isinstance(node.slice, ast.Tuple):
            parts = [_annotation_name(element) for element in node.slice.elts]
            if all(part is not None for part in parts):
                return f"{base}[{','.join(parts)}]"  # type: ignore[arg-type]
        slice_name = _annotation_name(node.slice)
        if slice_name is not None:
            return f"{base}[{slice_name}]"
    return None


def _collect_violations(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    if "type: ignore" in source:
        return [f"{path.name}: contains type: ignore comment"]
    tree = ast.parse(source, filename=str(path))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _FORBIDDEN_IMPORTS:
                    violations.append(f"{path.name}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split(".", 1)[0] in _FORBIDDEN_IMPORTS:
                violations.append(f"{path.name}:{node.lineno} imports {node.module}")
            if node.module == "typing":
                for alias in node.names:
                    if alias.name in _FORBIDDEN_TYPING_NAMES:
                        violations.append(f"{path.name}:{node.lineno} imports typing.{alias.name}")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                violations.append(f"{path.name}:{node.lineno} calls {node.func.id}()")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
            key = node.value.id
            if isinstance(node.slice, ast.Tuple) and len(node.slice.elts) == 2:
                left = _annotation_name(node.slice.elts[0])
                right = _annotation_name(node.slice.elts[1])
                if (key, right) in _FORBIDDEN_SUBSCRIPT_KEYS or (left, right) == ("str", "object"):
                    violations.append(
                        f"{path.name}:{node.lineno} uses forbidden annotation {key}[{left}, {right}]"
                    )
            slice_name = _annotation_name(node.slice)
            if key == "Mapping" and slice_name == "str,object":
                violations.append(f"{path.name}:{node.lineno} uses Mapping[str, object]")
        for field in ("annotation", "returns"):
            child = getattr(node, field, None)
            if child is not None:
                name = _annotation_name(child)
                if name == "object":
                    violations.append(f"{path.name}:{node.lineno} annotates object")
    return violations


@pytest.mark.parametrize("relative_path", _SCOPED_PRODUCTION_FILES)
def test_vector_index_administration_control_plane_has_no_weak_typing(
    relative_path: str,
) -> None:
    path = _repo_root() / relative_path
    violations = _collect_violations(path)
    assert violations == [], "\n".join(violations)


def test_generic_vector_index_contract_has_no_qdrant_vendor_imports() -> None:
    path = _repo_root() / _SCOPED_PRODUCTION_FILES[0]
    source = path.read_text(encoding="utf-8")
    assert "qdrant_client" not in source
    assert "qdrant" not in source.casefold()
