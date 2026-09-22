# © Artur Czarnecki. All rights reserved.

"""Deterministic AST scan for canonical OBS/DIAG authority constructor sites."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from testing_support.obs_diag_freeze.manifest import CANONICAL_AUTHORITY_OWNERS

_PRODUCTION_SCAN_ROOTS = (
    "intergrax/applications",
    "intergrax/runtime",
    "agents",
    "applications",
    "scripts",
)

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "tests",
        "docker",
        "runtime-context",
        "node_modules",
    }
)

_SYMBOL_TO_ALLOWED: dict[str, frozenset[str]] = {
    owner.symbol: frozenset(owner.direct_construction_allowed_modules)
    for owner in CANONICAL_AUTHORITY_OWNERS
}


@dataclass(frozen=True, slots=True)
class AuthorityConstructorViolation:
    rel_path: str
    lineno: int
    symbol: str
    change_class: str


def _is_allowed_construction(rel_posix: str, symbol: str) -> bool:
    allowed = _SYMBOL_TO_ALLOWED.get(symbol)
    if allowed is None:
        return True
    return rel_posix in allowed


def _call_symbol(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def collect_authority_constructor_violations_in_source(
    source: str,
    *,
    rel_path: str,
) -> tuple[AuthorityConstructorViolation, ...]:
    tree = ast.parse(source, filename=rel_path)
    violations: list[AuthorityConstructorViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        symbol = _call_symbol(node)
        if symbol is None or symbol not in _SYMBOL_TO_ALLOWED:
            continue
        if _is_allowed_construction(rel_path, symbol):
            continue
        violations.append(
            AuthorityConstructorViolation(
                rel_path=rel_path,
                lineno=node.lineno,
                symbol=symbol,
                change_class="requalification_required",
            ),
        )
    return tuple(violations)


def collect_production_authority_constructor_violations(
    repo_root: Path,
) -> tuple[AuthorityConstructorViolation, ...]:
    violations: list[AuthorityConstructorViolation] = []
    for scan_root in _PRODUCTION_SCAN_ROOTS:
        base = repo_root / scan_root
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            rel = path.relative_to(repo_root).as_posix()
            try:
                source = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            try:
                chunk = collect_authority_constructor_violations_in_source(source, rel_path=rel)
            except SyntaxError:
                continue
            violations.extend(chunk)
    return tuple(violations)
