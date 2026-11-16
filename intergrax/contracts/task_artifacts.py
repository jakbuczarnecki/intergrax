# © Artur Czarnecki. All rights reserved.

"""Task artifact references shared by runtime completion and application hosts.

Single definition (ADR-PDC-08, PDC-D8-R017): ``intergrax.applications.contracts.application_artifacts``
is a COMPAT-X shim that re-exports these classes.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.contracts.artifact_ref import (
    ArtifactClass,
    ArtifactDigestRequiredError,
    ArtifactLineage,
    ArtifactProvenance,
    ArtifactRetentionClass,
    ArtifactScope,
    DurableArtifactRef,
)
from intergrax.contracts.run_artifact_bundle import RUN_ARTIFACT_BUNDLE_METADATA_KEY

APPLICATION_ARTIFACTS_STAGING_KEY: Final[str] = "application_artifacts.v1"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_sha256_hex(value: str) -> str:
    """Accept the legacy empty value (digest unknown) or a lowercase hex sha256, nothing else."""
    if value and not _SHA256_HEX_RE.match(value):
        raise ValueError("sha256 must be empty (unknown) or 64 lowercase hex characters")
    return value


def _durable_digest(artifact_id: str, sha256: str) -> str:
    if not sha256:
        raise ArtifactDigestRequiredError(
            f"artifact {artifact_id!r} has no sha256; a durable reference requires a content digest"
        )
    return f"sha256:{sha256}"


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

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_hex(value)


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

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_hex(value)


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

    @field_validator("sha256")
    @classmethod
    def _check_sha256(cls, value: str) -> str:
        return _validate_sha256_hex(value)


class RunArtifactBundle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "run_artifact_bundle.v1"
    task_id: str
    graph_id: str | None = None
    run_id: str | None = None
    application: list[ApplicationArtifactRef] = Field(default_factory=list)
    workspace: list[WorkspaceArtifactRef] = Field(default_factory=list)
    sandbox: list[SandboxArtifactRef] = Field(default_factory=list)

    def durable_refs(self, *, run_id: str | None = None) -> tuple[DurableArtifactRef, ...]:
        """Project the rollup onto the shared durable reference contract (producer, lineage, digest).

        An entry without a content digest raises ``ArtifactDigestRequiredError`` instead of
        yielding an empty digest. ``run_id`` defaults to the bundle's own ``run_id``.
        """
        effective_run = run_id or self.run_id
        refs: list[DurableArtifactRef] = []
        for item in self.application:
            refs.append(
                DurableArtifactRef(
                    artifact_id=item.artifact_id,
                    artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
                    type=item.kind,
                    uri=item.uri,
                    mime_type=item.content_type,
                    content_digest=_durable_digest(item.artifact_id, item.sha256),
                    size_bytes=item.size_bytes,
                    scope=ArtifactScope(tenant_id=item.tenant_id),
                    producer=ArtifactProvenance(created_by_agent_id=item.owner_app_id),
                    lineage=ArtifactLineage(
                        run_id=item.run_id or effective_run or item.task_id,
                        task_id=item.task_id,
                    ),
                    retention_class=_retention_class(item.retention),
                    created_at=item.created_at,
                )
            )
        for ws in self.workspace:
            refs.append(
                DurableArtifactRef(
                    artifact_id=ws.artifact_id,
                    artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
                    type="workspace_file",
                    uri=ws.uri,
                    content_digest=_durable_digest(ws.artifact_id, ws.sha256),
                    size_bytes=ws.size_bytes,
                    scope=ArtifactScope(tenant_id=ws.tenant_id, workspace_id=ws.workspace_id),
                    producer=ArtifactProvenance(created_by_agent_id=f"workspace:{ws.workspace_id}"),
                    lineage=ArtifactLineage(run_id=effective_run or ws.task_id, task_id=ws.task_id),
                    retention_class=_retention_class(ws.retention),
                )
            )
        for sb in self.sandbox:
            refs.append(
                DurableArtifactRef(
                    artifact_id=sb.artifact_id,
                    artifact_class=ArtifactClass.EXECUTION_EVIDENCE,
                    type="sandbox_file",
                    uri=sb.uri,
                    content_digest=_durable_digest(sb.artifact_id, sb.sha256),
                    size_bytes=sb.size_bytes,
                    scope=ArtifactScope(tenant_id=sb.tenant_id),
                    producer=ArtifactProvenance(
                        created_by_agent_id=f"sandbox:{sb.session_id}",
                        created_by_tool_id=sb.tool_id,
                    ),
                    lineage=ArtifactLineage(run_id=effective_run or sb.task_id, task_id=sb.task_id),
                    retention_class=_retention_class(sb.retention),
                )
            )
        return tuple(refs)


def _retention_class(policy: ArtifactRetentionPolicy) -> ArtifactRetentionClass:
    """Map the legacy retention directive onto a retention class (no TTL is decided here)."""
    if policy.retain_hours is not None and policy.retain_hours > 24:
        return ArtifactRetentionClass.EXTENDED
    if policy.delete_on_task_complete:
        return ArtifactRetentionClass.EPHEMERAL
    return ArtifactRetentionClass.DEFAULT


__all__ = [
    "APPLICATION_ARTIFACTS_STAGING_KEY",
    "RUN_ARTIFACT_BUNDLE_METADATA_KEY",
    "ApplicationArtifactRef",
    "ArtifactRetentionPolicy",
    "ArtifactSecurityClass",
    "ArtifactVisibility",
    "RunArtifactBundle",
    "SandboxArtifactRef",
    "WorkspaceArtifactRef",
]
