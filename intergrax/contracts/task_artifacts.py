# © Artur Czarnecki. All rights reserved.

"""Task artifact references shared by runtime completion and application hosts."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

APPLICATION_ARTIFACTS_STAGING_KEY: Final[str] = "application_artifacts.v1"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ArtifactSecurityClass(StrEnum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ArtifactVisibility(StrEnum):
    TASK_ONLY = "task_only"
    APPLICATION = "application"
    TENANT = "tenant"
    OPERATOR = "operator"


class ArtifactRetentionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    retain_hours: int | None = Field(default=None, ge=1)
    delete_on_task_complete: bool = True
    archive_to_object_store: bool = False


class ApplicationArtifactRef(BaseModel):
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
        default_factory=lambda: ArtifactRetentionPolicy(delete_on_task_complete=True, retain_hours=24),
    )


class RunArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "run_artifact_bundle.v1"
    task_id: str
    graph_id: str | None = None
    application: list[ApplicationArtifactRef] = Field(default_factory=list)
    workspace: list[WorkspaceArtifactRef] = Field(default_factory=list)
    sandbox: list[SandboxArtifactRef] = Field(default_factory=list)
