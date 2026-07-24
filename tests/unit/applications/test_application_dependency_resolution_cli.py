# © Artur Czarnecki. All rights reserved.

"""Manual uv resolver isolation verification for application dependencies."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
LKW = "local_workspace_application"
NON_SLACK = "lab_application"


@pytest.mark.gate
@pytest.mark.no_ci
def test_dependency_tree_isolation_slack_sdk() -> None:
    """Resolver-level isolation: LKW includes slack-sdk; lab does not."""

    def export_for(project: str) -> str:
        proc = subprocess.run(
            [
                "uv",
                "export",
                "--frozen",
                "--no-dev",
                "--project",
                f"applications/{project}",
                "--no-emit-workspace",
            ],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.lower()

    lkw = export_for(LKW)
    lab = export_for(NON_SLACK)
    assert "slack-sdk" in lkw
    assert "slack-sdk" not in lab
