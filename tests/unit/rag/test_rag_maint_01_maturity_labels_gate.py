# © Artur Czarnecki. All rights reserved.

"""RAG-MAINT-01 — canonical maturity label gate must run without import cycles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPT = _REPO_ROOT / "scripts" / "maintenance" / "check_rag_maturity_labels.py"


def test_rag_maint_01_maturity_labels_script_executes() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "rag maturity label audit: OK" in completed.stdout
