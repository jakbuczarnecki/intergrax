# © Artur Czarnecki. All rights reserved.

"""Architecture gate — RAG multichannel retrieval must not depend on scenario proofs."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_IMPORT_ROOTS = (
    "platform_proofs",
    "platform_proofs.scenarios.verified_product_identification",
)


def _iter_python_files(root: Path):
    for path in root.rglob("*.py"):
        yield path


def test_multichannel_retrieval_has_no_platform_proofs_imports() -> None:
    package_root = (
        Path(__file__).resolve().parents[4]
        / "intergrax"
        / "rag"
        / "retrieval"
        / "multichannel"
    )
    violations: list[str] = []
    for path in _iter_python_files(package_root):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_forbidden(alias.name):
                        violations.append(f"{path}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if _is_forbidden(node.module):
                    violations.append(f"{path}: from {node.module}")
    assert violations == []


def _is_forbidden(module_name: str) -> bool:
    return any(
        module_name == root or module_name.startswith(f"{root}.")
        for root in FORBIDDEN_IMPORT_ROOTS
    )
