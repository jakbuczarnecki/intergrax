# © Artur Czarnecki. All rights reserved.

"""Optional Docker image build for poc_template_application (not in default gate)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POC_PKG = "poc_template_application"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_poc_template_dockerfile_builds() -> None:
    """Build reference Tier-3 image via minimal runtime-graph context."""
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/build/build_application_image.py",
        "--application",
        _POC_PKG,
        "--tag",
        "intergrax-poc-template-test:gate",
    ]
    result = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]
