#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Static secret-literal gate over the product tree (F-D6-14 / PDC-D6-R007, T-IGX-051-B Part H).

Runs the existing ``scan_tree_for_raw_secrets`` scanner and a second, script-owned AST pass for the
escape shapes the plain scanner misses:

- string concatenation of constants (``"sk-" + "..."``), folded before matching;
- f-strings whose parts are all constants, folded before matching;
- bytes literals with a credential format (or bound to a secret-named target);
- ``"".join([...])`` of constant parts (a split token prefix);
- ``x or "literal"`` and ``if not x: x = "literal"`` fallbacks bound to a secret-named target.

Findings carry path, line, rule and the identifier, never the value. Exit 1 on any finding that
is not in the script-owned allowlist (path, rule, identifier, reason). Usage::

    uv run python scripts/maintenance/check_static_secret_literals.py [--root DIR] [--list-allowed]
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from intergrax.core.security import raw_secret_gate
from intergrax.core.security.raw_secret_gate import (
    KNOWN_FORBIDDEN_LITERALS,
    RawSecretFinding,
    iter_python_files,
    scan_tree_for_raw_secrets,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_ROOTS: tuple[str, ...] = ("intergrax", "agents", "applications")

RULE_FOLDED_CONCATENATION = "folded_concatenation_literal"
RULE_CONSTANT_FSTRING = "constant_fstring_literal"
RULE_BYTES_LITERAL = "bytes_secret_literal"
RULE_JOINED_PREFIX = "joined_prefix_literal"
RULE_OR_FALLBACK = "or_literal_fallback"
RULE_IF_NOT_FALLBACK = "if_not_literal_fallback"


@dataclass(frozen=True, slots=True)
class AllowedFinding:
    """One triaged false positive: matched by (path, rule, identifier), with the reason."""

    path: str
    rule: str
    name: str
    reason: str


# Script-owned allowlist of triaged false positives (step S0-10). A true positive is never listed
# here: it stays a finding until the product code is fixed.
ALLOWLIST: tuple[AllowedFinding, ...] = (
    AllowedFinding(
        path="intergrax/agent_distribution/control_plane_governance.py",
        rule=RULE_OR_FALLBACK,
        name="revision_token",
        reason="serving-pointer revision placeholder (a revision id token, not a credential)",
    ),
    AllowedFinding(
        path="intergrax/core/security/raw_secret_gate.py",
        rule=RULE_JOINED_PREFIX,
        name="literal",
        reason="the gate's own known-forbidden literal, deliberately built from parts",
    ),
)


def _credential_shaped(value: str) -> bool:
    return value in KNOWN_FORBIDDEN_LITERALS or any(p.search(value) for p in raw_secret_gate._CREDENTIAL_FORMATS)


def _is_secret_name(name: str) -> bool:
    return raw_secret_gate._is_secret_name(name)


def _looks_like_named_secret(value: str) -> bool:
    return raw_secret_gate._looks_like_named_secret(value)


def _fold(node: ast.AST | None) -> str | None:
    """Constant string value of ``node`` after folding concatenation, f-strings, bytes and joins."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return node.value
        if isinstance(node.value, bytes):
            return node.value.decode("latin-1")
        return None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            folded = _fold(value.value if isinstance(value, ast.FormattedValue) else value)
            if folded is None:
                return None
            parts.append(folded)
        return "".join(parts)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "join"
        and len(node.args) == 1
        and not node.keywords
        and isinstance(node.args[0], (ast.List, ast.Tuple))
    ):
        separator = _fold(node.func.value)
        items = [_fold(item) for item in node.args[0].elts]
        if separator is None or any(item is None for item in items):
            return None
        return separator.join(item for item in items if item is not None)
    return None


def _shape_rule(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, bytes):
        return RULE_BYTES_LITERAL
    if isinstance(node, ast.BinOp):
        return RULE_FOLDED_CONCATENATION
    if isinstance(node, ast.JoinedStr):
        return RULE_CONSTANT_FSTRING
    if isinstance(node, ast.Call):
        return RULE_JOINED_PREFIX
    return None


def _target_names(node: ast.AST) -> Iterator[str]:
    if isinstance(node, ast.Name):
        yield node.id
    elif isinstance(node, ast.Attribute):
        yield node.attr
    elif isinstance(node, (ast.Tuple, ast.List)):
        for item in node.elts:
            yield from _target_names(item)
    elif isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
        yield node.slice.value


def _fallback_literal(node: ast.AST | None) -> str | None:
    """The trailing literal of ``x or "literal"``, when it is a non-empty, non-variable-name string."""
    if not isinstance(node, ast.BoolOp) or not isinstance(node.op, ast.Or):
        return None
    literal = _fold(node.values[-1])
    if not literal or not literal.strip() or raw_secret_gate._ENV_VAR_LIKE.match(literal):
        return None
    return literal


def scan_source_for_evasions(source: str, *, path: str) -> tuple[RawSecretFinding, ...]:
    """Second pass over one module: the escape shapes; syntax errors yield no findings."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()
    findings: list[RawSecretFinding] = []

    def add(node: ast.AST, rule: str, name: str) -> None:
        findings.append(RawSecretFinding(path=path, line=getattr(node, "lineno", 0), rule=rule, name=name))

    bound: dict[int, list[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bound[id(node.value)] = [n for t in node.targets for n in _target_names(t)]
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            bound[id(node.value)] = list(_target_names(node.target))
        elif isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg:
                    bound[id(keyword.value)] = [keyword.arg]

    for node in ast.walk(tree):
        rule = _shape_rule(node)
        if rule is not None:
            folded = _fold(node)
            if folded is not None:
                names = [n for n in bound.get(id(node), []) if _is_secret_name(n)]
                if _credential_shaped(folded):
                    add(node, rule, names[0] if names else "literal")
                elif names and _looks_like_named_secret(folded):
                    add(node, rule, names[0])
        literal = _fallback_literal(node)
        if literal is not None:
            for name in (n for n in bound.get(id(node), []) if _is_secret_name(n)):
                add(node, RULE_OR_FALLBACK, name)
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.UnaryOp)
            and isinstance(node.test.op, ast.Not)
            and isinstance(node.test.operand, ast.Name)
            and _is_secret_name(node.test.operand.id)
        ):
            target = node.test.operand.id
            for statement in node.body:
                if (
                    isinstance(statement, ast.Assign)
                    and target in [n for t in statement.targets for n in _target_names(t)]
                    and _fold(statement.value)
                ):
                    add(statement, RULE_IF_NOT_FALLBACK, target)
    return tuple(findings)


def _allowed(finding: RawSecretFinding, allowlist: Sequence[AllowedFinding]) -> bool:
    return any(
        item.path == finding.path and item.rule == finding.rule and item.name == finding.name for item in allowlist
    )


def scan_product_tree(
    root: Path,
    *,
    product_roots: Iterable[str] = PRODUCT_ROOTS,
    allowlist: Sequence[AllowedFinding] = ALLOWLIST,
) -> tuple[RawSecretFinding, ...]:
    """Both passes over ``root/<product_root>``; allowlisted findings are removed; sorted, unique."""
    roots = [root / name for name in product_roots]
    findings = set(scan_tree_for_raw_secrets(roots, relative_to=root))
    for product_root in roots:
        if not product_root.exists():
            continue
        for file_path, relative in iter_python_files(product_root, relative_to=root):
            try:
                text = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            findings.update(scan_source_for_evasions(text, path=relative))
    kept = [finding for finding in findings if not _allowed(finding, allowlist)]
    return tuple(sorted(kept, key=lambda item: (item.path, item.line, item.rule, item.name)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="repository root to scan")
    parser.add_argument("--list-allowed", action="store_true", help="print the allowlist and exit")
    args = parser.parse_args(argv)
    if args.list_allowed:
        for item in ALLOWLIST:
            print(f"{item.path}:{item.rule}:{item.name}  # {item.reason}")
        return 0
    findings = scan_product_tree(args.root)
    for finding in findings:
        print(f"{finding.path}:{finding.line}: {finding.rule} ({finding.name})")
    if findings:
        print(f"FAIL: {len(findings)} static secret-literal finding(s); values are never printed", file=sys.stderr)
        return 1
    print("OK: no static secret literal in the product tree")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
