# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R2-R1-R1 — canonical tool definitions at runtime LLM dispatch boundary."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]

_PRODUCTION_ROOTS = (
    _REPO_ROOT / "intergrax/runtime",
    _REPO_ROOT / "agents",
    _REPO_ROOT / "applications",
)

_SKIP_DIR_NAMES = frozenset({"tests", "testing_support", "__pycache__"})
_SKIP_PATH_PARTS = frozenset({"proofs"})

_TOOL_DISPATCH_METHODS = frozenset({"generate_with_tools", "stream_with_tools"})

_ALLOWED_TOOLS_ARG_NAMES = frozenset(
    {
        "canonical_tool_definitions",
        "canonical_tools",
        "tool_definitions",
        "definitions",
    }
)

_FORBIDDEN_DISPATCH_UNION_MARKERS = (
    "CanonicalFunctionToolDefinition | Mapping",
    "Mapping[str, object]",
    "Mapping[str, JsonValue]",
    "dict[str, object]",
)


def _iter_production_python_files() -> list[Path]:
    paths: list[Path] = []
    for root in _PRODUCTION_ROOTS:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            if any(part in _SKIP_DIR_NAMES for part in path.parts):
                continue
            if any(part in _SKIP_PATH_PARTS for part in path.parts):
                continue
            rel = path.relative_to(_REPO_ROOT).as_posix()
            if "docker/runtime-context" in rel:
                continue
            if "/llm_adapters/" in f"/{rel}/":
                continue
            paths.append(path)
    return sorted(paths)


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def _is_materialize_call(node: ast.expr) -> bool:
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            return func.id == "materialize_canonical_tool_definitions_for_llm_dispatch"
        if isinstance(func, ast.Attribute):
            return func.attr == "materialize_canonical_tool_definitions_for_llm_dispatch"
    return False


def _tools_argument_is_canonical(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id in _ALLOWED_TOOLS_ARG_NAMES
    if _is_materialize_call(node):
        return True
    if isinstance(node, ast.Tuple) and len(node.elts) == 1:
        return _tools_argument_is_canonical(node.elts[0])
    return False


def _is_adapter_tool_dispatch_call(node: ast.Call) -> bool:
    if not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr not in _TOOL_DISPATCH_METHODS:
        return False
    base = node.func.value
    if isinstance(base, ast.Name) and base.id in {"self", "cls"}:
        return False
    return True


def test_ebh_2e_r2_r1_r1_tool_planning_service_has_no_raw_mapping_dispatch_union() -> None:
    path = _REPO_ROOT / "intergrax/runtime/nexus/tools/tool_planning_service.py"
    source = path.read_text(encoding="utf-8-sig")
    assert "provider_tools" not in source
    dispatch_union_offenders = [
        line.strip()
        for line in source.splitlines()
        if "canonical_tool_definitions" in line
        and any(marker in line for marker in _FORBIDDEN_DISPATCH_UNION_MARKERS)
    ]
    assert not dispatch_union_offenders, "\n".join(dispatch_union_offenders)


def test_ebh_2e_r2_r1_r1_production_llm_tool_dispatch_uses_canonical_materialization() -> None:
    offenders: list[str] = []
    for path in _iter_production_python_files():
        try:
            tree = _module_ast(path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_adapter_tool_dispatch_call(node):
                continue
            if len(node.args) < 2:
                continue
            tools_arg = node.args[1]
            if not _tools_argument_is_canonical(tools_arg):
                rel = path.relative_to(_REPO_ROOT).as_posix()
                offenders.append(f"{rel}:{node.lineno} tools arg not canonical-bound")
    assert not offenders, "\n".join(offenders)


def test_ebh_2e_r2_r1_r1_runtime_canonical_materialization_owner_exists() -> None:
    path = _REPO_ROOT / "intergrax/runtime/nexus/tools/canonical_tool_dispatch.py"
    assert path.is_file()
    tree = _module_ast(path)
    defined = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "materialize_canonical_tool_definitions_for_llm_dispatch" in defined
