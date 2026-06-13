# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.gate, pytest.mark.no_ci]

import importlib
import subprocess
import sys
from pathlib import Path


def test_langgraph_audit_script_passes() -> None:
    repo = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts" / "check_langgraph_not_required.py")],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_harness_imports_without_langgraph() -> None:
    importlib.import_module("intergrax.harness")
    importlib.import_module("intergrax.runtime.nexus.nexus_loop")
