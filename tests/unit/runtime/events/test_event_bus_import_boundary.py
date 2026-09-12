# © Artur Czarnecki. All rights reserved.

"""W5-H — RuntimeEventBus ↔ event_delivery cold-start import boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from intergrax.runtime.events.event_bus import RuntimeEventBus

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _run_import_subprocess(statement: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", statement],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_event_bus_standalone_import() -> None:
    completed = _run_import_subprocess("import intergrax.runtime.events.event_bus")
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_event_delivery_then_event_bus_import_order() -> None:
    completed = _run_import_subprocess(
        "import intergrax.runtime.observability.event_delivery; "
        "import intergrax.runtime.events.event_bus",
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_event_bus_construct_and_close() -> None:
    bus = RuntimeEventBus()
    bus.close()
    assert bus.closed
