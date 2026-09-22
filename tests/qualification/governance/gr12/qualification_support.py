# © Artur Czarnecki. All rights reserved.

"""Shared helpers for GR-12 control-plane surface qualification tests."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _function_names_in_test_module(module_path: Path) -> set[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def assert_proof_nodes_registered(node_ids: tuple[str, ...]) -> None:
    missing: list[str] = []
    for node_id in node_ids:
        rel, name = node_id.split("::", 1)
        module_path = _REPO_ROOT / rel
        if not module_path.is_file():
            missing.append(node_id)
            continue
        if name not in _function_names_in_test_module(module_path):
            missing.append(node_id)
    assert missing == [], f"missing proof tests: {missing}"
