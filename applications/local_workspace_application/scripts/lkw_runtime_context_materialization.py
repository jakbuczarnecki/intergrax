# © Artur Czarnecki. All rights reserved.

"""Shared LKW application-image runtime-context materialization."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

LOCAL_WORKSPACE_APPLICATION = "local_workspace_application"
MATERIALIZE_ONLY_FLAG = "--materialize-only"


class RuntimeContextMaterializationError(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def materialize_local_workspace_runtime_context(
    *,
    repo_root: Path,
    application_image_builder: Path,
    runtime_context_dir: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]],
    timeout_seconds: int = 300,
) -> None:
    completed = run_command(
        [
            "uv",
            "run",
            "python",
            str(application_image_builder),
            "--application",
            LOCAL_WORKSPACE_APPLICATION,
            "--context-dir",
            str(runtime_context_dir),
            MATERIALIZE_ONLY_FLAG,
        ],
        cwd=repo_root,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeContextMaterializationError("runtime_context_materialization_failed")
