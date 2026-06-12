# © Artur Czarnecki. All rights reserved.

"""Immutable environment materialization for deploy and task intake (APP-EVOL-1 · §49.1.2)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.task.task_metadata_keys import TaskMetadataKey

ENV_SNAPSHOT_RUNTIME_KEY = TaskMetadataKey.ENVIRONMENT_SNAPSHOT


class SnapshotCaptureSource(StrEnum):
    """How an :class:`EnvironmentSnapshot` was produced."""

    DEPLOY = "deploy"
    INTAKE = "intake"
    MANUAL_EXPORT = "manual_export"


class EnvironmentSnapshot(BaseModel):
    """
    Immutable materialization of everything Nexus needs for one deploy or Task intake.

    Wire format key: ``environment_snapshot.v1`` inside ``HookContext.runtime_state``
    and ``Task.metadata``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = "environment_snapshot.v1"
    snapshot_id: str
    app_id: str
    app_version: str
    profile_snapshot_id: str
    manifest_digest: str
    graph_spec_digest: str | None = None
    org_envelope_digest: str | None = None
    roster_digest: str
    captured_at: str = Field(description="UTC ISO-8601 timestamp")
    captured_by: SnapshotCaptureSource = SnapshotCaptureSource.INTAKE
