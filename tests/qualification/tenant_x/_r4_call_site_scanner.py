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

CallSiteResolution = Literal["resolved", "ambiguous_star_import", "unresolved"]

ScanFailureKind = Literal["read_failure", "parse_failure"]


@dataclass(frozen=True, slots=True)
class R4DiscoveredProductionCallSite:
    module: str
    containing_symbol: str
    callee: ProtectedCallee
    line_no: int
    resolution: CallSiteResolution


@dataclass(frozen=True, slots=True)
class R4ModuleScanFailure:
    module: str
    kind: ScanFailureKind
    detail: str


@dataclass(frozen=True, slots=True)
class R4ProductionScanResult:
    eligible_modules: tuple[str, ...]
    successfully_scanned: tuple[str, ...]
    failures: tuple[R4ModuleScanFailure, ...]
    call_sites: tuple[R4DiscoveredProductionCallSite, ...]


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
    """Maps local names to module path roots for attribute chaining."""

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
                if alias.asname:
                    local = alias.asname
                    by_name[local] = alias.name
                else:
                    root = alias.name.split(".")[0]
                    by_name[root] = root
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
) -> tuple[ProtectedCallee | None, CallSiteResolution]:
    if isinstance(func, ast.Name):
        symbol = func.id
        qualified = bindings.by_name.get(symbol)
        if qualified is not None:
            for callee in _PROTECTED_DEFINING_MODULE:
                if qualified == _protected_qualified_name(callee):
                    return callee, "resolved"
            return None, "unresolved"
        if symbol not in _PROTECTED_DEFINING_MODULE:
            return None, "unresolved"
        if _PROTECTED_DEFINING_MODULE[symbol] == module_rel:
            return symbol, "resolved"
        if star_import:
            return symbol, "ambiguous_star_import"
        return symbol, "unresolved"
    if isinstance(func, ast.Attribute):
        symbol = func.attr
        if symbol not in _PROTECTED_DEFINING_MODULE:
            return None, "unresolved"
        expected_mod = module_name_from_rel(_PROTECTED_DEFINING_MODULE[symbol])
        if isinstance(func.value, ast.Name):
            mod_path = bindings.by_name.get(func.value.id)
            if mod_path is None:
                if star_import:
                    return symbol, "ambiguous_star_import"
                return symbol, "unresolved"
            if mod_path == expected_mod:
                return symbol, "resolved"
            return None, "unresolved"
        dotted = _dotted_module_from_expr(func.value, bindings)
        if dotted == expected_mod:
            return symbol, "resolved"
        if dotted is None:
            if star_import:
                return symbol, "ambiguous_star_import"
            return symbol, "unresolved"
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


def discover_call_sites_in_module(
    module_rel: str,
    source: str,
) -> tuple[R4DiscoveredProductionCallSite, ...]:
    """Parse module source and return protected call-site candidates (raises SyntaxError on parse failure)."""
    tree = ast.parse(source, filename=module_rel)
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


def r4_scan_accounting_holds(result: R4ProductionScanResult) -> bool:
    return len(result.eligible_modules) == len(result.successfully_scanned) + len(result.failures)


def r4_unresolved_protected_call_sites(
    result: R4ProductionScanResult,
) -> tuple[R4DiscoveredProductionCallSite, ...]:
    return tuple(s for s in result.call_sites if s.resolution != "resolved")


@lru_cache(maxsize=1)
def scan_r4_production_modules() -> R4ProductionScanResult:
    eligible: list[str] = []
    success: list[str] = []
    failures: list[R4ModuleScanFailure] = []
    sites: list[R4DiscoveredProductionCallSite] = []
    if not _INTERGRAX_ROOT.is_dir():
        return R4ProductionScanResult((), (), (), ())

    for path in sorted(_INTERGRAX_ROOT.rglob("*.py")):
        if not _is_production_intergrax_file(path):
            continue
        rel = _normalize_repo_path(path)
        eligible.append(rel)
        try:
            source = path.read_text(encoding="utf-8")
        except OSError as exc:
            failures.append(
                R4ModuleScanFailure(module=rel, kind="read_failure", detail=str(exc))
            )
            continue
        try:
            sites.extend(discover_call_sites_in_module(rel, source))
        except SyntaxError as exc:
            failures.append(
                R4ModuleScanFailure(module=rel, kind="parse_failure", detail=str(exc))
            )
            continue
        success.append(rel)

    return R4ProductionScanResult(
        eligible_modules=tuple(eligible),
        successfully_scanned=tuple(success),
        failures=tuple(failures),
        call_sites=tuple(sites),
    )


@lru_cache(maxsize=1)
def discover_r4_protected_production_call_sites() -> tuple[R4DiscoveredProductionCallSite, ...]:
    return scan_r4_production_modules().call_sites


def discover_r4_calls_for_callee(callee: ProtectedCallee) -> tuple[R4DiscoveredProductionCallSite, ...]:
    return tuple(s for s in discover_r4_protected_production_call_sites() if s.callee == callee)
