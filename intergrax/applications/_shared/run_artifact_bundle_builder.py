# © Artur Czarnecki. All rights reserved.

"""Compatibility re-export for application-layer artifact staging."""

from __future__ import annotations

from intergrax.contracts.task_artifacts import (
    APPLICATION_ARTIFACTS_STAGING_KEY,
    ApplicationArtifactRef,
    RunArtifactBundle,
    SandboxArtifactRef,
    WorkspaceArtifactRef,
)
from intergrax.runtime.nexus.orchestration.run_artifact_bundle_builder import build_run_artifact_bundle
from intergrax.runtime.task.task import Task


def stage_application_artifact(task: Task, ref: ApplicationArtifactRef) -> None:
    """Append a host-produced artifact ref to task metadata for final rollup."""
    raw = task.metadata.get(APPLICATION_ARTIFACTS_STAGING_KEY, [])
    staged: list[dict[str, object]] = []
    if isinstance(raw, list):
        staged = [item for item in raw if isinstance(item, dict)]
    staged.append(ref.model_dump(mode="json"))
    task.metadata[APPLICATION_ARTIFACTS_STAGING_KEY] = staged
    task.sync_metadata()


__all__ = [
    "APPLICATION_ARTIFACTS_STAGING_KEY",
    "ApplicationArtifactRef",
    "RunArtifactBundle",
    "SandboxArtifactRef",
    "WorkspaceArtifactRef",
    "build_run_artifact_bundle",
    "stage_application_artifact",
]
