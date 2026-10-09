# © Artur Czarnecki. All rights reserved.

"""AST/static helpers for TRACE-X-P5-R2-P4 qualification gates."""

from __future__ import annotations

import ast
from pathlib import Path

TRACE_X_P5_R2_P4_START_HEAD = "5c47154066915820cd150d4fd370bc8b1ae8f85c"

_REPO_ROOT = Path(__file__).resolve().parents[3]

RECONSTRUCTOR_MODULE = (
    _REPO_ROOT / "intergrax/runtime/observability/reconstruction/execution_reconstruction.py"
)
PROJECTION_MODULE = (
    _REPO_ROOT
    / "intergrax/runtime/observability/reconstruction/integration_configuration_provenance_projection.py"
)
READER_MODULE = (
    _REPO_ROOT
    / "intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py"
)

FORBIDDEN_RECONSTRUCTOR_IMPORT_PREFIXES = (
    "intergrax.integrations.existing_capability_configuration",
    "intergrax.applications._shared.integrations.persistence",
    "intergrax.integrations.registry",
    "intergrax.integrations.execution_bound_integration_resolution",
)

FORBIDDEN_PROJECTION_IMPORT_PREFIXES = (
    "intergrax.applications._shared.integrations",
    "intergrax.integrations.providers",
    "intergrax.integrations.registry",
    "intergrax.integrations.existing_capability_configuration",
)


def _read_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def module_import_prefix_violations(path: Path, forbidden: tuple[str, ...]) -> list[str]:
    tree = _read_ast(path)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for prefix in forbidden:
                if node.module.startswith(prefix):
                    violations.append(f"{path.name}: import from {node.module}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in forbidden:
                    if alias.name.startswith(prefix):
                        violations.append(f"{path.name}: import {alias.name}")
    return violations


def count_class_definitions(path: Path, class_name: str) -> int:
    tree = _read_ast(path)
    return sum(
        1
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
