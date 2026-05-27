# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.runtime.workspace.manager import ShadowWorkspaceManager
from intergrax.runtime.workspace.shadow_workspace import ShadowWorkspace

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_shadow_workspace_write_read_and_manifest(tmp_path):
    workspace = ShadowWorkspace.create(
        tmp_path,
        tenant_id="t1",
        task_id="task-1",
    )
    artifact = workspace.write_text("notes/summary.txt", "hello shadow")
    assert artifact.relative_path == "notes/summary.txt"
    assert workspace.read_text("notes/summary.txt") == "hello shadow"

    manifest = workspace.manifest()
    assert manifest.artifact_count == 1
    assert manifest.artifacts[0].sha256 == artifact.sha256


def test_shadow_workspace_snapshot_and_rollback(tmp_path):
    workspace = ShadowWorkspace.create(
        tmp_path,
        tenant_id="t1",
        task_id="task-2",
    )
    workspace.write_text("v1.txt", "version one")
    snapshot = workspace.snapshot()

    workspace.write_text("v1.txt", "version two")
    workspace.write_text("extra.txt", "extra")
    assert workspace.read_text("v1.txt") == "version two"

    workspace.rollback(snapshot)
    assert workspace.read_text("v1.txt") == "version one"
    assert {artifact.relative_path for artifact in workspace.list_artifacts()} == {"v1.txt"}


def test_shadow_workspace_rejects_unsafe_paths(tmp_path):
    workspace = ShadowWorkspace.create(
        tmp_path,
        tenant_id="t1",
        task_id="task-3",
    )
    with pytest.raises(ValueError):
        workspace.write_text("../escape.txt", "nope")


def test_shadow_workspace_manager_cleanup(tmp_path):
    manager = ShadowWorkspaceManager(root=tmp_path)
    workspace = manager.open_or_create(tenant_id="t1", task_id="task-4")
    workspace.write_text("data.txt", "persist")
    assert workspace.exists_on_disk()

    assert manager.cleanup(workspace.workspace_id) is True
    assert not workspace.exists_on_disk()
    assert manager.get(workspace.workspace_id) is None
