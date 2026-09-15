# © Artur Czarnecki. All rights reserved.

"""Import-cycle regression guard for runtime event export mapping."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_RUNTIME_EVENT_DELIVERY = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "event_delivery"
    / "runtime_event_delivery.py"
)
_RUNTIME_EVENT_EXPORT_MAPPING = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "runtime_event_export_mapping.py"
)
_RUNTIME_EVENT_EXPORT_MODELS = (
    _REPO_ROOT
    / "intergrax"
    / "runtime"
    / "observability"
    / "runtime_event_export_models.py"
)
_FORBIDDEN_IMPORT_MODULES = frozenset(
    {
        "intergrax.runtime.observability.export_boundary",
        "intergrax.runtime.observability.event_delivery.runtime_event_delivery",
    }
)


def _module_must_not_import(
    path: Path,
    *,
    forbidden_modules: frozenset[str],
    subject: str,
) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module in forbidden_modules:
            pytest.fail(
                f"{subject} must not import {node.module} (line {node.lineno})",
            )


def test_observability_facade_cold_import_in_subprocess() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from intergrax.runtime.observability import InMemoryObservabilityExporter",
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_event_delivery_does_not_import_export_boundary() -> None:
    _module_must_not_import(
        _RUNTIME_EVENT_DELIVERY,
        forbidden_modules=frozenset(
            {"intergrax.runtime.observability.export_boundary"},
        ),
        subject="runtime_event_delivery",
    )


def test_runtime_event_export_mapping_does_not_import_export_boundary() -> None:
    _module_must_not_import(
        _RUNTIME_EVENT_EXPORT_MAPPING,
        forbidden_modules=frozenset(
            {"intergrax.runtime.observability.export_boundary"},
        ),
        subject="runtime_event_export_mapping",
    )


def test_runtime_event_export_models_neutral_imports() -> None:
    _module_must_not_import(
        _RUNTIME_EVENT_EXPORT_MODELS,
        forbidden_modules=_FORBIDDEN_IMPORT_MODULES,
        subject="runtime_event_export_models",
    )


def test_export_boundary_and_mapper_import_order_independence() -> None:
    scripts = (
        (
            "from intergrax.runtime.observability.runtime_event_export_mapping "
            "import runtime_event_export_source_from_event; "
            "from intergrax.runtime.observability.export_boundary "
            "import RuntimeEventExportSource"
        ),
        (
            "from intergrax.runtime.observability.export_boundary "
            "import RuntimeEventExportSource; "
            "from intergrax.runtime.observability.runtime_event_export_mapping "
            "import runtime_event_export_source_from_event"
        ),
    )
    for script in scripts:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=_REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
