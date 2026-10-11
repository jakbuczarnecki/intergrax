# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Explicit legacy JSON readers for pre-v1 Agent Distribution SQLite payloads (COMPAT-X-R2-R2)."""

from __future__ import annotations

import json
from datetime import datetime
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.deployment import (
    SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1,
    DeploymentInstanceRecord,
    DeploymentInstanceState,
)
from intergrax.agent_distribution.runtime_materialization_record import (
    SCHEMA_RUNTIME_MATERIALIZATION_RECORD_V1,
    RuntimeMaterializationRecord,
)
from intergrax.agent_distribution.runtime_revision import MaterializationTopology
from intergrax.agent_distribution.stores import (
    SCHEMA_AGENT_ARTIFACT_METADATA_V1,
    SCHEMA_APPLICATION_ENVIRONMENT_SERVING_RECORD_V1,
    AgentArtifactMetadata,
    ApplicationEnvironmentServingRecord,
)

LEGACY_UNVERSIONED_PRE_V1 = "LEGACY_UNVERSIONED_PRE_V1"

_NON_EMPTY = Field(min_length=1)
LegacyT = TypeVar("LegacyT", bound=BaseModel)
CurrentT = TypeVar("CurrentT", bound=BaseModel)


def _mapping_without_schema_version(data: object) -> dict[str, object] | None:
    if not isinstance(data, dict):
        return None
    if "schema_version" in data:
        return None
    return data


class _LegacyAgentArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    package_digest: str = _NON_EMPTY
    artifact_store_ref: str = Field(min_length=1)
    distribution_package_id: str = Field(min_length=1)
    agent_project_metadata_ref: str = _NON_EMPTY
    tombstoned: bool = False

    @field_validator("agent_project_metadata_ref")
    @classmethod
    def _strip_metadata_ref(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)


class _LegacyRuntimeMaterializationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    materialization_topology: MaterializationTopology
    artifact_locator: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    materialized_runtime_lock_id: str = _NON_EMPTY
    materialized_runtime_lock_digest: str = _NON_EMPTY


class _LegacyDeploymentInstanceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    instance_state: DeploymentInstanceState
    readiness_evidence_ref: str | None = None
    serving_unit_ref: str | None = None
    prepared_at: datetime | None = None
    ready_at: datetime | None = None
    drain_started_at: datetime | None = None
    drain_completed_at: datetime | None = None
    failure_evidence_ref: str | None = None
    record_revision: int = Field(default=0, ge=0)


class _LegacyApplicationEnvironmentServingRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    traffic_serving_revision_id: str | None = None
    serving_pointer_revision: int = Field(default=0, ge=0)
    prior_traffic_revision_id: str | None = None
    committed_at: datetime | None = None


def _decode_versioned_or_legacy(
    payload: str,
    *,
    expected_schema: str,
    current_type: type[CurrentT],
    legacy_type: type[LegacyT],
    migrate: Callable[[LegacyT], CurrentT],
    family_label: str,
) -> CurrentT:
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"corrupt {family_label} payload: expected JSON object")
    schema_version = data.get("schema_version")
    if schema_version == expected_schema:
        return current_type.model_validate(data)
    legacy_mapping = _mapping_without_schema_version(data)
    if legacy_mapping is not None:
        legacy = legacy_type.model_validate(legacy_mapping)
        return migrate(legacy)
    raise ValueError(
        f"unsupported schema version {schema_version!r} for {family_label}; "
        f"expected {expected_schema!r} or {LEGACY_UNVERSIONED_PRE_V1}"
    )


def decode_agent_artifact_metadata_payload(payload: str) -> AgentArtifactMetadata:
    def _migrate(legacy: _LegacyAgentArtifactMetadata) -> AgentArtifactMetadata:
        return AgentArtifactMetadata(
            schema_version=SCHEMA_AGENT_ARTIFACT_METADATA_V1,
            **legacy.model_dump(),
        )

    return _decode_versioned_or_legacy(
        payload,
        expected_schema=SCHEMA_AGENT_ARTIFACT_METADATA_V1,
        current_type=AgentArtifactMetadata,
        legacy_type=_LegacyAgentArtifactMetadata,
        migrate=_migrate,
        family_label="AgentArtifactMetadata",
    )


def decode_runtime_materialization_record_payload(payload: str) -> RuntimeMaterializationRecord:
    def _migrate(legacy: _LegacyRuntimeMaterializationRecord) -> RuntimeMaterializationRecord:
        return RuntimeMaterializationRecord(
            schema_version=SCHEMA_RUNTIME_MATERIALIZATION_RECORD_V1,
            **legacy.model_dump(),
        )

    return _decode_versioned_or_legacy(
        payload,
        expected_schema=SCHEMA_RUNTIME_MATERIALIZATION_RECORD_V1,
        current_type=RuntimeMaterializationRecord,
        legacy_type=_LegacyRuntimeMaterializationRecord,
        migrate=_migrate,
        family_label="RuntimeMaterializationRecord",
    )


def decode_deployment_instance_record_payload(payload: str) -> DeploymentInstanceRecord:
    def _migrate(legacy: _LegacyDeploymentInstanceRecord) -> DeploymentInstanceRecord:
        return DeploymentInstanceRecord(
            schema_version=SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1,
            **legacy.model_dump(),
        )

    return _decode_versioned_or_legacy(
        payload,
        expected_schema=SCHEMA_DEPLOYMENT_INSTANCE_RECORD_V1,
        current_type=DeploymentInstanceRecord,
        legacy_type=_LegacyDeploymentInstanceRecord,
        migrate=_migrate,
        family_label="DeploymentInstanceRecord",
    )


def decode_application_environment_serving_record_payload(
    payload: str,
) -> ApplicationEnvironmentServingRecord:
    def _migrate(
        legacy: _LegacyApplicationEnvironmentServingRecord,
    ) -> ApplicationEnvironmentServingRecord:
        return ApplicationEnvironmentServingRecord(
            schema_version=SCHEMA_APPLICATION_ENVIRONMENT_SERVING_RECORD_V1,
            **legacy.model_dump(),
        )

    return _decode_versioned_or_legacy(
        payload,
        expected_schema=SCHEMA_APPLICATION_ENVIRONMENT_SERVING_RECORD_V1,
        current_type=ApplicationEnvironmentServingRecord,
        legacy_type=_LegacyApplicationEnvironmentServingRecord,
        migrate=_migrate,
        family_label="ApplicationEnvironmentServingRecord",
    )


__all__ = [
    "LEGACY_UNVERSIONED_PRE_V1",
    "decode_agent_artifact_metadata_payload",
    "decode_application_environment_serving_record_payload",
    "decode_deployment_instance_record_payload",
    "decode_runtime_materialization_record_payload",
]
