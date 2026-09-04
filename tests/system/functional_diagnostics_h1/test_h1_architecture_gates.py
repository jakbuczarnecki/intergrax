# © Artur Czarnecki. All rights reserved.

"""Static architecture gates owned by DIAG-FUNCTIONAL-H1."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_ANALYZER_PATH = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics" / "functional_diagnostic_analyzer.py"
_FORBIDDEN_TELEMETRY_PREFIXES = (
    "opentelemetry",
    "otel",
    "langfuse",
    "sentry_sdk",
    "datadog",
)
_FORBIDDEN_QUEUE_PREFIXES = (
    "celery",
    "kafka",
    "rabbitmq",
    "intergrax.queueing",
)


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def test_analyzer_does_not_import_vendor_telemetry() -> None:
    imports = _module_imports(_ANALYZER_PATH)
    for module in imports:
        lowered = module.lower()
        for forbidden in _FORBIDDEN_TELEMETRY_PREFIXES:
            assert forbidden not in lowered, f"analyzer imports telemetry module {module}"


def test_diagnostics_runtime_does_not_import_queue_delivery() -> None:
    diagnostics_dir = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
    violations: list[str] = []
    for path in diagnostics_dir.rglob("*.py"):
        for module in _module_imports(path):
            lowered = module.lower()
            for forbidden in _FORBIDDEN_QUEUE_PREFIXES:
                if forbidden in lowered:
                    violations.append(f"{path.relative_to(_REPO_ROOT)} imports {module}")
    assert violations == [], "; ".join(violations)


def test_h1_package_has_no_dynamic_imports() -> None:
    h1_dir = _REPO_ROOT / "tests" / "system" / "functional_diagnostics_h1"
    violations: list[str] = []
    for path in h1_dir.glob("*.py"):
        if path.name == "test_h1_architecture_gates.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "__import__":
                        violations.append(f"{path.name}: dynamic __import__")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in {"eval", "exec", "__import__"}:
                    violations.append(f"{path.name}:{node.lineno}: {node.func.id}")
    assert violations == [], "; ".join(violations)
