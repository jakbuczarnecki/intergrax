# © Artur Czarnecki. All rights reserved.

"""Regression: HARDENING-2 self-healing SPI typing (pyright Protocol stubs)."""

from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]

_SPI_PROTOCOL_METHODS: tuple[tuple[str, str], ...] = (
    ("intergrax/contracts/self_healing/observation/provider.py", "observe"),
    ("intergrax/contracts/self_healing/selection/selector.py", "select"),
    ("intergrax/contracts/self_healing/validation/validator.py", "validate"),
    ("intergrax/contracts/self_healing/workflow/rollback.py", "plan_rollback"),
    ("intergrax/contracts/self_healing/workflow/validation.py", "validate"),
)


def _protocol_method_ends_with_ellipsis(source_path: Path, method_name: str) -> bool:
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(base, ast.Name) and base.id == "Protocol" for base in node.bases):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == method_name:
                if not item.body:
                    return False
                last = item.body[-1]
                return isinstance(last, ast.Expr) and isinstance(last.value, ast.Constant) and (
                    last.value.value is ...
                )
    raise AssertionError(f"Protocol method {method_name!r} not found in {source_path}")


def test_self_healing_spi_protocol_methods_end_with_ellipsis_stub() -> None:
    """Protocol bodies must not be docstring-only (pyright reportReturnType)."""
    for rel_path, method_name in _SPI_PROTOCOL_METHODS:
        path = _REPO_ROOT / rel_path
        assert _protocol_method_ends_with_ellipsis(path, method_name), (
            f"{rel_path}:{method_name} must end with ``...`` stub"
        )
