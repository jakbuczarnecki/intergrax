# © Artur Czarnecki. All rights reserved.

"""Architecture gate — direct OpenTelemetry SDK imports are allowlisted (HARDEN-3E)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_ALLOWED_RELATIVE_PREFIXES = (
    "intergrax/rag/tracking/",
    "intergrax/context/tracking/",
    "intergrax/integrations/providers/observability_backend/",
)

_SCAN_ROOTS = (
    "intergrax",
    "agents",
    "applications",
)

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "tests",
    }
)

_OTEL_IMPORT_MODULES = frozenset(
    {
        "opentelemetry",
        "opentelemetry.trace",
        "opentelemetry.sdk",
        "opentelemetry.sdk.trace",
        "opentelemetry.sdk.trace.export",
        "opentelemetry.sdk.trace.export.in_memory_span_exporter",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _is_allowed_path(path: Path) -> bool:
    rel = path.relative_to(_repo_root()).as_posix()
    return any(rel.startswith(prefix) for prefix in _ALLOWED_RELATIVE_PREFIXES)


def _production_python_files() -> list[Path]:
    root = _repo_root()
    files: list[Path] = []
    for scan_root in _SCAN_ROOTS:
        base = root / scan_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if any(part in _EXCLUDED_PARTS for part in path.parts):
                continue
            if _is_allowed_path(path):
                continue
            files.append(path)
    return files


def _collect_direct_otel_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    violations: list[str] = []
    rel = path.relative_to(_repo_root()).as_posix()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".", 1)[0]
                if root_module == "opentelemetry":
                    violations.append(f"{rel}:{node.lineno} imports {alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            root_module = node.module.split(".", 1)[0]
            if root_module == "opentelemetry":
                violations.append(f"{rel}:{node.lineno} imports from {node.module}")
    return violations


def test_direct_opentelemetry_imports_are_allowlisted() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        violations.extend(_collect_direct_otel_imports(path))
    assert violations == []
