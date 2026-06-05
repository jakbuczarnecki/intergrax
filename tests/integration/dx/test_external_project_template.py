# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


@pytest.mark.gate
def test_external_project_template_py_compile() -> None:
    template = Path(__file__).resolve().parents[3] / "intergrax" / "scaffold" / "external_project"
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "demo_harness"
        shutil.copytree(template, target)
        app_py = target / "app.py"
        proc = subprocess.run(
            [sys.executable, "-m", "py_compile", str(app_py)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr or proc.stdout
