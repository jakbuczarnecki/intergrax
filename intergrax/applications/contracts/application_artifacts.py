# © Artur Czarnecki. All rights reserved.

"""Application artifact references — provenance and retention (APP-CON-4)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

RUN_ARTIFACT_BUNDLE_METADATA_KEY: Final[str] = "run_artifact_bundle.v1"
APPLICATION_ARTIFACTS_STAGING_KEY: Final[str] = "application_artifacts.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactSecurityClass(StrEnum):
    """Classification for access policy and retention."""

    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ArtifactVisibility(StrEnum):
    """Who may read the artifact outside the owning task."""

    TASK_ONLY = "task_only"
    APPLICATION = "application"
    TENANT = "tenant"
    OPERATOR = "operator"


class ArtifactRetentionPolicy(BaseModel):
    """Retention directive for environment artifacts."""

    model_config = ConfigDict(extra="forbid")

    retain_hours: int | None = Field(default=None, ge=1)
    delete_on_task_complete: bool = True
    archive_to_object_store: bool = False


class ApplicationArtifactRef(BaseModel):
    """Generic application-level artifact produced during a Task."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    kind: str
    uri: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0
    sha256: str = ""
    task_id: str
    run_id: str | None = None
    graph_id: str | None = None
    owner_app_id: str
    tenant_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    provenance: str = "application"
    security_class: ArtifactSecurityClass = ArtifactSecurityClass.INTERNAL
    visibility: ArtifactVisibility = ArtifactVisibility.TASK_ONLY
    retention: ArtifactRetentionPolicy = Field(default_factory=ArtifactRetentionPolicy)


class WorkspaceArtifactRef(BaseModel):
    """Shadow workspace artifact with workspace provenance."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    workspace_id: str
    relative_path: str
    uri: str
    size_bytes: int = 0
    sha256: str = ""
    task_id: str
    tenant_id: str
    snapshot_id: str | None = None
    security_class: ArtifactSecurityClass = ArtifactSecurityClass.INTERNAL
    visibility: ArtifactVisibility = ArtifactVisibility.TASK_ONLY
    retention: ArtifactRetentionPolicy = Field(default_factory=ArtifactRetentionPolicy)


class SandboxArtifactRef(BaseModel):
    """Sandbox execution output artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact_id: str
    session_id: str
    relative_path: str
    uri: str
    size_bytes: int = 0
    sha256: str = ""
    task_id: str
    tenant_id: str
    tool_id: str | None = None
    security_class: ArtifactSecurityClass = ArtifactSecurityClass.RESTRICTED
    visibility: ArtifactVisibility = ArtifactVisibility.TASK_ONLY
    retention: ArtifactRetentionPolicy = Field(
        default_factory=lambda: ArtifactRetentionPolicy(delete_on_task_complete=True, retain_hours=24)
    )


class RunArtifactBundle(BaseModel):
    """Rollup linked from ``ApplicationRunSummary`` metadata (Plane A)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "run_artifact_bundle.v1"
    task_id: str
    graph_id: str | None = None
    application: list[ApplicationArtifactRef] = Field(default_factory=list)
    workspace: list[WorkspaceArtifactRef] = Field(default_factory=list)
    sandbox: list[SandboxArtifactRef] = Field(default_factory=list)
