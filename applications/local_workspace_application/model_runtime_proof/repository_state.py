# © Artur Czarnecki. All rights reserved.

"""Git working-tree capture for truthful proof metadata."""

from __future__ import annotations

import subprocess
from pathlib import Path

from local_workspace_application.model_runtime_proof.contracts import (
    RepositoryStateRecord,
)

_TASK_OWNED_PREFIXES: tuple[str, ...] = (
    "applications/local_workspace_application/model_runtime_proof/",
    "applications/local_workspace_application/tests/model_runtime/",
    "applications/local_workspace_application/tests/e2e/test_model_runtime_portability_live.py",
    "applications/local_workspace_application/scripts/run-lkw-model-runtime-proof.py",
    "applications/local_workspace_application/docs/evidence/LKW_MODEL_RUNTIME_PORTABILITY.",
    "infra/docker/vllm/docker-compose.yml",
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_task_owned(path: str) -> bool:
    normalized = _normalize_path(path)
    return any(
        normalized.startswith(prefix) or normalized == prefix.rstrip("/")
        for prefix in _TASK_OWNED_PREFIXES
    )


def _git_head(repo_root: Path | None = None) -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return output.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_porcelain(repo_root: Path | None = None) -> list[str]:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in output.splitlines() if line.strip()]


def _dirty_path_from_porcelain(line: str) -> str | None:
    if len(line) < 4:
        return None
    path = line[3:].strip()
    if " -> " in path:
        path = path.split(" -> ", 1)[1].strip()
    return _normalize_path(path)


def capture_repository_state(
    *,
    repo_root: Path | None = None,
) -> RepositoryStateRecord:
    head = _git_head(repo_root)
    lines = _git_porcelain(repo_root)
    task_owned: list[str] = []
    unrelated: list[str] = []
    for line in lines:
        path = _dirty_path_from_porcelain(line)
        if not path:
            continue
        if _is_task_owned(path):
            task_owned.append(path)
        else:
            unrelated.append(path)

    if not lines:
        classification = "clean"
    elif task_owned and unrelated:
        classification = "task_owned_and_unrelated_changes"
    elif task_owned:
        classification = "task_owned_changes"
    elif unrelated:
        classification = "unrelated_changes"
    else:
        classification = "unavailable"

    return RepositoryStateRecord(
        repository_head_at_proof=head,
        repository_head_role="pre_evidence_commit_head",
        working_tree_classification=classification,
        task_owned_dirty_paths=tuple(sorted(task_owned)),
        unrelated_dirty_paths=tuple(sorted(unrelated)),
    )
