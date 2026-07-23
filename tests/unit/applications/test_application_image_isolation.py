# © Artur Czarnecki. All rights reserved.

"""Representative Docker image isolation proofs (LKW + Lab)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

_DOCKER = shutil.which("docker")


def _docker_available() -> bool:
    if not _DOCKER:
        return False
    proc = subprocess.run(
        ["docker", "info"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


pytestmark = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)


def _build(app: str, tag: str) -> None:
    proc = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/build/build_application_image.py",
            "--application",
            app,
            "--tag",
            tag,
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    assert '"application"' in proc.stdout or '"schema_version"' in proc.stdout


def _run(tag: str, script: str) -> str:
    proc = subprocess.run(
        ["docker", "run", "--rm", tag, "python", "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    return proc.stdout


@pytest.mark.integration
def test_lkw_image_filesystem_and_deps() -> None:
    tag = "intergrax/local-workspace:isolation-test"
    _build("local_workspace_application", tag)  # noqa: intentional side effect
    out = _run(
        tag,
        ";".join(
            [
                "from pathlib import Path",
                "apps=[p.name for p in Path('/app/applications').iterdir() if p.is_dir()]",
                "agents=sorted(p.name for p in Path('/app/agents').iterdir() if p.is_dir())",
                "import slack_sdk, pymongo, sentry_sdk",
                "print('APPS='+','.join(apps))",
                "print('AGENTS='+','.join(agents))",
                "print('VENV='+str(Path('/app/.venv').is_dir()))",
            ]
        ),
    )
    assert "APPS=local_workspace_application" in out
    assert "legal_application" not in out
    assert "lab_application" not in out
    for agent in ("local_indexer", "local_search", "local_synthesizer"):
        assert agent in out
    assert "legal" not in out.split("AGENTS=", 1)[-1]
    assert "VENV=True" in out


@pytest.mark.integration
def test_lab_image_excludes_slack_and_other_apps() -> None:
    tag = "intergrax/lab:isolation-test"
    _build("lab_application", tag)
    out = _run(
        tag,
        ";".join(
            [
                "from pathlib import Path",
                "import importlib.util",
                "apps=[p.name for p in Path('/app/applications').iterdir() if p.is_dir()]",
                "agents=sorted(p.name for p in Path('/app/agents').iterdir() if p.is_dir())",
                "print('APPS='+','.join(apps))",
                "print('AGENTS='+','.join(agents))",
                "print('SLACK='+str(importlib.util.find_spec('slack_sdk') is not None))",
                "print('VENV='+str(Path('/app/.venv').is_dir()))",
            ]
        ),
    )
    assert "APPS=lab_application" in out
    assert "local_workspace_application" not in out
    assert "local_indexer" not in out
    assert "SLACK=False" in out
    assert "VENV=True" in out
