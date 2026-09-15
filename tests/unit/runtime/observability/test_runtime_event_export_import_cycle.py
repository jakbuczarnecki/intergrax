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
    tree = ast.parse(
        _RUNTIME_EVENT_DELIVERY.read_text(encoding="utf-8"),
        filename=str(_RUNTIME_EVENT_DELIVERY),
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.module == "intergrax.runtime.observability.export_boundary":
            pytest.fail(
                "runtime_event_delivery must not import export_boundary "
                f"(line {node.lineno})",
            )
