# © Artur Czarnecki. All rights reserved.

"""Ensure DIST-SCALE-X P1 does not introduce shadow scheduler/dispatcher/retry authority."""

from __future__ import annotations

import ast
from pathlib import Path

_FORBIDDEN_DEFINITION_NAMES = frozenset(
    {
        "TaskScheduler",
        "BackgroundDispatcher",
        "RetryAuthority",
        "ShadowScheduler",
        "QualificationScheduler",
    },
)

_FORBIDDEN_IMPORT_SUFFIXES = (
    "queueing.worker.dispatcher",
    "queueing.worker.retry_policy",
)


def scan_dist_scale_x_authority_surface(package_dir: Path) -> tuple[str, ...]:
    violations: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        if path.name.startswith("test_"):
            continue
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name in _FORBIDDEN_DEFINITION_NAMES:
                violations.append(f"{path.name}: forbidden class {node.name}")
            if isinstance(node, ast.FunctionDef) and node.name in _FORBIDDEN_DEFINITION_NAMES:
                violations.append(f"{path.name}: forbidden function {node.name}")
            if isinstance(node, ast.AsyncFunctionDef) and node.name in _FORBIDDEN_DEFINITION_NAMES:
                violations.append(f"{path.name}: forbidden async function {node.name}")
            if isinstance(node, ast.ImportFrom) and node.module:
                for suffix in _FORBIDDEN_IMPORT_SUFFIXES:
                    if node.module.endswith(suffix) or node.module == suffix:
                        violations.append(f"{path.name}: forbidden import from {node.module}")
    return tuple(violations)
