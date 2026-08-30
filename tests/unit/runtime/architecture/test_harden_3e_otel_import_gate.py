# © Artur Czarnecki. All rights reserved.

"""Architecture gate — direct OpenTelemetry SDK imports are allowlisted (HARDEN-3E)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_DIRECT_OTEL_ALLOWED_FILES = frozenset(
    {
        "intergrax/rag/tracking/rag_spans.py",
        "intergrax/context/tracking/context_spans.py",
    }
)

_SCAN_ROOTS = (
    "intergrax",
    "agents",
    "applications",
)

_EXCLUDED_PARTS = frozenset(
    {
        "__pycache__",
        "runtime-context",
        "tests",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


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
            files.append(path)
    return files


def _collect_direct_otel_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
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
    root = _repo_root()
    for path in _production_python_files():
        rel = path.relative_to(root).as_posix()
        direct_imports = _collect_direct_otel_imports(path)
        if not direct_imports:
            continue
        if rel not in _DIRECT_OTEL_ALLOWED_FILES:
            violations.extend(direct_imports)
    assert violations == []


def test_allowlisted_otel_files_exist() -> None:
    root = _repo_root()
    missing = sorted(
        rel for rel in _DIRECT_OTEL_ALLOWED_FILES if not (root / rel).is_file()
    )
    assert missing == []


def test_new_tracking_module_not_auto_allowlisted() -> None:
    assert "intergrax/rag/tracking/new_otel_module.py" not in _DIRECT_OTEL_ALLOWED_FILES
    assert "intergrax/context/tracking/new_otel_module.py" not in _DIRECT_OTEL_ALLOWED_FILES
