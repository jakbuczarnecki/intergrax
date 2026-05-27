# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Lifecycle manager for active shadow workspaces."""

from __future__ import annotations

import os
from pathlib import Path

from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace

ENV_SHADOW_ROOT = "INTERGRAX_SHADOW_ROOT"
DEFAULT_SHADOW_ROOT = Path("build") / "shadow_workspaces"


def resolve_shadow_root(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    env = os.environ.get(ENV_SHADOW_ROOT, "").strip()
    if env:
        return Path(env)
    return DEFAULT_SHADOW_ROOT


class ShadowWorkspaceManager:
    """Creates, tracks, and cleans up shadow workspaces per tenant/task."""

    def __init__(self, *, root: Path | None = None) -> None:
        self._root = resolve_shadow_root(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._active: dict[str, ShadowWorkspace] = {}

    @property
    def root(self) -> Path:
        return self._root

    def open_or_create(self, *, tenant_id: str, task_id: str) -> ShadowWorkspace:
        key = f"{tenant_id}:{task_id}"
        existing = self._active.get(key)
        if existing is not None and existing.exists_on_disk():
            return existing

        workspace = ShadowWorkspace.create(
            self._root,
            tenant_id=tenant_id,
            task_id=task_id,
        )
        self._active[key] = workspace
        self._active[workspace.workspace_id] = workspace
        return workspace

    def get(self, workspace_id: str) -> ShadowWorkspace | None:
        return self._active.get(workspace_id)

    def cleanup(self, workspace_id: str) -> bool:
        workspace = self._active.pop(workspace_id, None)
        if workspace is None:
            return False
        key = f"{workspace.tenant_id}:{workspace.task_id}"
        self._active.pop(key, None)
        workspace.cleanup()
        return True

    def cleanup_for_task(self, *, tenant_id: str, task_id: str) -> bool:
        key = f"{tenant_id}:{task_id}"
        workspace = self._active.pop(key, None)
        if workspace is None:
            return False
        self._active.pop(workspace.workspace_id, None)
        workspace.cleanup()
        return True
