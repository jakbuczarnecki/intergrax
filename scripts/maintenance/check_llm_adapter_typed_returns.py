#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""CI guard: LLMAdapter public methods must not return bare str or Dict[str, Any]."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADAPTER_ROOT = ROOT / "intergrax" / "llm_adapters"
CONTRACT_FILE = ADAPTER_ROOT / "contracts" / "llm_adapter.py"

FORBIDDEN_PUBLIC_RETURNS = frozenset({"str", "Dict[str, Any]", "dict"})


def _annotation_name(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Subscript):
        base = _annotation_name(node.value)
        slice_node = node.slice
        if isinstance(slice_node, ast.Tuple):
            parts = [_annotation_name(elt) for elt in slice_node.elts]
            inner = ", ".join(p or "?" for p in parts)
        else:
            inner = _annotation_name(slice_node) or "?"
        return f"{base or '?'}[{inner}]"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        left = _annotation_name(node.left) or "?"
        right = _annotation_name(node.right) or "?"
        return f"{left} | {right}"
    return ast.unparse(node)


def _collect_llm_adapter_methods() -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(CONTRACT_FILE.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "LLMAdapter":
            out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[item.name] = item
            return out
    raise RuntimeError("LLMAdapter class not found in contracts/llm_adapter.py")


def main() -> int:
    methods = _collect_llm_adapter_methods()
    public = {
        name: fn
        for name, fn in methods.items()
        if not name.startswith("_")
        and name
        in {
            "generate_messages",
            "stream_messages",
            "generate_with_tools",
            "stream_with_tools",
            "generate_structured",
        }
    }
    violations: list[str] = []
    for name, fn in public.items():
        ret = _annotation_name(fn.returns)
        if ret in FORBIDDEN_PUBLIC_RETURNS:
            violations.append(f"{name} -> {ret}")

    if violations:
        print("LLM adapter contract uses forbidden public return types:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        return 1

    print("check_llm_adapter_typed_returns: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
