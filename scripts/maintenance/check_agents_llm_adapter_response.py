#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CI guard: Tier-2 agents must consume LLMAdapterResponse, not bare str (M-LLM-R.6.4)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGENTS_ROOT = ROOT / "agents"

ADAPTER_METHODS = frozenset({"generate_messages", "generate_with_tools", "stream_messages"})


def _type_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _type_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        base = _type_name(node.value) or "?"
        inner = _type_name(node.slice) or "?"
        return f"{base}[{inner}]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _type_name(node.left) or "?"
        right = _type_name(node.right) or "?"
        return f"{left} | {right}"
    return ast.unparse(node)


def _collect_violations(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix()
    source = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [f"{rel}: syntax error: {exc}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name not in ADAPTER_METHODS:
                continue
            ret = _type_name(node.returns)
            if ret in {"str", "Dict", "dict"} or ret == "Dict[str, Any]":
                violations.append(
                    f"{rel}:{node.lineno}: {node.name} -> {ret} (must return LLMAdapterResponse or LLMStreamEvent)"
                )
            continue

        if not isinstance(node, ast.AnnAssign):
            continue
        if not isinstance(node.target, ast.Name):
            continue
        if _type_name(node.annotation) != "str":
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if isinstance(func, ast.Attribute) and func.attr in ADAPTER_METHODS:
            violations.append(
                f"{rel}:{node.lineno}: {node.target.id}: str = adapter.{func.attr}(...) "
                "— use LLMAdapterResponse and .content"
            )

    return violations


def main() -> int:
    if not AGENTS_ROOT.is_dir():
        print("check_agents_llm_adapter_response: agents/ not found", file=sys.stderr)
        return 1

    all_violations: list[str] = []
    for path in sorted(AGENTS_ROOT.rglob("*.py")):
        if "tests" in path.parts:
            continue
        all_violations.extend(_collect_violations(path))

    if all_violations:
        print("Tier-2 agents assume bare str from LLM adapter:", file=sys.stderr)
        for item in all_violations:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print("check_agents_llm_adapter_response: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
