# © Artur Czarnecki. All rights reserved.

"""Optional Docker image build for poc_template_application (not in default gate)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POC_PKG = "poc_template_application"
_DOCKERFILE = _REPO_ROOT / "applications" / _POC_PKG / "docker" / "Dockerfile"
_IGNORE = _REPO_ROOT / "applications" / _POC_PKG / "docker" / ".dockerignore"

pytestmark = pytest.mark.integration


@pytest.mark.skipif(shutil.which("docker") is None, reason="docker CLI not available")
def test_poc_template_dockerfile_builds() -> None:
    """Build reference Tier-3 image from monorepo root (slow; run explicitly)."""
    cmd = [
        "docker",
        "build",
        "-f",
        str(_DOCKERFILE),
        "-t",
        "intergrax-poc-template-test:gate",
        str(_REPO_ROOT),
    ]
    if shutil.which("docker") and _has_buildx():
        cmd = [
            "docker",
            "buildx",
            "build",
            "-f",
            str(_DOCKERFILE),
            "--ignorefile",
            str(_IGNORE),
            "-t",
            "intergrax-poc-template-test:gate",
            "--load",
            str(_REPO_ROOT),
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, (result.stdout or "")[-2000:] + (result.stderr or "")[-2000:]


def _has_buildx() -> bool:
    proc = subprocess.run(
        ["docker", "buildx", "version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0
