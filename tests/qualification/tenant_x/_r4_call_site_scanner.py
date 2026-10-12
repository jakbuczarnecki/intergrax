# © Artur Czarnecki. All rights reserved.

"""AST mechanical discovery of R4 protected recording call-sites in intergrax/**/*.py."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final, Literal

_REPO_ROOT = Path(__file__).resolve().parents[3]
_INTERGRAX_ROOT = _REPO_ROOT / "intergrax"

_EXCLUDE_PATH_PARTS: Final[tuple[str, ...]] = (
    "/tests/",
    "\\tests\\",
    "/test_",
    "\\test_",
    "/qualification/",
    "\\qualification\\",
)

ProtectedCallee = Literal[
    "record_context_assembly",
    "record_llm_call_runtime_event",
    "maybe_record_llm_call_from_usage_end",
]

_PROTECTED_DEFINING_MODULE: Final[dict[ProtectedCallee, str]] = {
    "record_context_assembly": "intergrax/runtime/events/context_skill_recording.py",
    "record_llm_call_runtime_event": "intergrax/runtime/events/llm_call_recording.py",
    "maybe_record_llm_call_from_usage_end": "intergrax/runtime/events/llm_call_recording.py",
}


@dataclass(frozen=True, slots=True)
class R4DiscoveredProductionCallSite:
    module: str
    containing_symbol: str
    callee: ProtectedCallee
    line_no: int
    resolution: Literal["resolved", "ambiguous_star_import", "unresolved"]


def module_name_from_rel(rel: str) -> str:
    stem = rel.removesuffix(".py").replace("/", ".")
    if stem.endswith(".__init__"):
        return stem[: -len(".__init__")]
    return stem


def _normalize_repo_path(path: Path) -> str:
    return path.relative_to(_REPO_ROOT).as_posix()


def _is_production_intergrax_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    normalized = path.as_posix()
    for part in _EXCLUDE_PATH_PARTS:
        if part.replace("/", "\\") in normalized or part in normalized:
            return False
    return True


@dataclass
class _ImportBindings:
    """Maps local names to fully-qualified module paths (dotted)."""

    by_name: dict[str, str]
    star_modules: tuple[str, ...]


def _resolve_relative_module(current_module: str, module: str | None, level: int) -> str:
    if level == 0:
        return module or current_module
    parts = current_module.split(".")
    parts = parts[:-level]
    if module:
        parts.extend(module.split("."))
    return ".".join(p for p in parts if p)


def _collect_import_bindings(tree: ast.Module, current_module: str) -> _ImportBindings:
    by_name: dict[str, str] = {}
    star_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[-1]
                by_name[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.module is None and node.level == 0:
                continue
            base = _resolve_relative_module(current_module, node.module, node.level)
            if any(alias.name == "*" for alias in node.names):
                star_modules.append(base)
                continue
            for alias in node.names:
                local = alias.asname or alias.name
                by_name[local] = f"{base}.{alias.name}"
    return _ImportBindings(by_name=by_name, star_modules=tuple(star_modules))


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_symbol(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> str:
    cur: ast.AST | None = node
    while cur is not None:
        if isinstance(cur, ast.FunctionDef | ast.AsyncFunctionDef):
            return cur.name
        cur = parents.get(cur)
    return "<module>"


def _function_defining_protected_symbol(
    module_rel: str,
    func_name: str,
    callee: ProtectedCallee,
) -> bool:
    return (
        _PROTECTED_DEFINING_MODULE[callee] == module_rel
        and func_name == callee
    )


def _dotted_module_from_expr(expr: ast.expr, bindings: _ImportBindings) -> str | None:
    if isinstance(expr, ast.Name):
        return bindings.by_name.get(expr.id)
    if isinstance(expr, ast.Attribute):
        prefix = _dotted_module_from_expr(expr.value, bindings)
        if prefix is None:
            return None
        return f"{prefix}.{expr.attr}"
    return None


def _protected_qualified_name(callee: ProtectedCallee) -> str:
    mod = module_name_from_rel(_PROTECTED_DEFINING_MODULE[callee])
    return f"{mod}.{callee}"


def _resolve_protected_callee(
    func: ast.expr,
    bindings: _ImportBindings,
    *,
    star_import: bool,
    module_rel: str,
) -> tuple[ProtectedCallee | None, Literal["resolved", "ambiguous_star_import", "unresolved"]]:
    if isinstance(func, ast.Name):
        symbol = func.id
        if symbol not in _PROTECTED_DEFINING_MODULE:
            return None, "unresolved"
        if _PROTECTED_DEFINING_MODULE[symbol] == module_rel:
            return symbol, "resolved"
        qualified = bindings.by_name.get(symbol)
        if qualified is None:
            if star_import:
                return None, "ambiguous_star_import"
            return None, "unresolved"
        if qualified == _protected_qualified_name(symbol):
            return symbol, "resolved"
        return None, "unresolved"
    if isinstance(func, ast.Attribute):
        symbol = func.attr
        if symbol not in _PROTECTED_DEFINING_MODULE:
            return None, "unresolved"
        expected_mod = module_name_from_rel(_PROTECTED_DEFINING_MODULE[symbol])
        if isinstance(func.value, ast.Name):
            mod_path = bindings.by_name.get(func.value.id)
            if mod_path is None:
                if star_import:
                    return None, "ambiguous_star_import"
                return None, "unresolved"
            if mod_path == expected_mod:
                return symbol, "resolved"
            return None, "unresolved"
        dotted = _dotted_module_from_expr(func.value, bindings)
        if dotted == expected_mod:
            return symbol, "resolved"
        return None, "unresolved"
    return None, "unresolved"


def call_expression_at_line(source: str, line_no: int, module_rel: str) -> ast.Call | None:
    try:
        tree = ast.parse(source, filename=module_rel)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.lineno == line_no:
            return node
    return None


def _discover_in_module(module_rel: str, source: str) -> tuple[R4DiscoveredProductionCallSite, ...]:
    try:
        tree = ast.parse(source, filename=module_rel)
    except SyntaxError:
        return ()
    if not isinstance(tree, ast.Module):
        return ()

    current_module = module_name_from_rel(module_rel)
    bindings = _collect_import_bindings(tree, current_module)
    star_import = bool(bindings.star_modules)
    parents = _parent_map(tree)
    found: list[R4DiscoveredProductionCallSite] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee, resolution = _resolve_protected_callee(
            node.func,
            bindings,
            star_import=star_import,
            module_rel=module_rel,
        )
        if callee is None:
            continue
        containing = _enclosing_symbol(node, parents)
        if _function_defining_protected_symbol(module_rel, containing, callee):
            continue
        found.append(
            R4DiscoveredProductionCallSite(
                module=module_rel,
                containing_symbol=containing,
                callee=callee,
                line_no=node.lineno or 0,
                resolution=resolution,
            )
        )
    return tuple(found)


@lru_cache(maxsize=1)
def discover_r4_protected_production_call_sites() -> tuple[R4DiscoveredProductionCallSite, ...]:
    sites: list[R4DiscoveredProductionCallSite] = []
    if not _INTERGRAX_ROOT.is_dir():
        return ()
    for path in sorted(_INTERGRAX_ROOT.rglob("*.py")):
        if not _is_production_intergrax_file(path):
            continue
        rel = _normalize_repo_path(path)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sites.extend(_discover_in_module(rel, source))
    return tuple(sites)


def discover_r4_calls_for_callee(callee: ProtectedCallee) -> tuple[R4DiscoveredProductionCallSite, ...]:
    return tuple(s for s in discover_r4_protected_production_call_sites() if s.callee == callee)
