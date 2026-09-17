# © Artur Czarnecki. All rights reserved.

"""AST-only detection of legacy ThreadedExecutionAdapter production imports (HOST-01)."""

from __future__ import annotations

import ast

THREADED_ADAPTER_MODULE = "intergrax.fastapi_core.execution.adapters.threaded_adapter"


def threaded_adapter_import_violations(tree: ast.AST, rel_path: str) -> list[str]:
    """Return human-readable violation lines for legacy threaded adapter wiring in one module."""
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                if mod == THREADED_ADAPTER_MODULE or mod.endswith(".threaded_adapter"):
                    violations.append(f"{rel_path}: import {mod!r}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            if mod == THREADED_ADAPTER_MODULE or mod.endswith(".threaded_adapter"):
                violations.append(f"{rel_path}: from {mod!r}")
            if any(
                isinstance(child, ast.alias) and child.name == "ThreadedExecutionAdapter"
                for child in node.names
            ):
                violations.append(f"{rel_path}: imports ThreadedExecutionAdapter from {mod!r}")
    return violations


def threaded_adapter_import_violations_from_source(source: str, rel_path: str) -> list[str]:
    try:
        tree = ast.parse(source, filename=rel_path)
    except SyntaxError:
        return []
    return threaded_adapter_import_violations(tree, rel_path)
