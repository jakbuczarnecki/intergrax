# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Runtime revision identity and lifecycle contracts (AGENT_DISTRIBUTION §18)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution.trust import AgentTrustEvidenceRef

_NON_EMPTY = Field(min_length=1)

SCHEMA_RUNTIME_REVISION_V1: Final = "runtime_revision.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RuntimeRevisionState(StrEnum):
    """Runtime revision lifecycle (§18.2)."""

    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    FAILED = "failed"


class MaterializationTopology(StrEnum):
    """Supported materialization topologies (§19.2)."""

    OCI_IMAGE = "oci_image"
    VENV_BUNDLE = "venv_bundle"
    SANDBOX_SIDECAR = "sandbox_sidecar"


_ACTIVE_OR_VALIDATED = frozenset(
    {
        RuntimeRevisionState.VALIDATED,
        RuntimeRevisionState.ACTIVE,
        RuntimeRevisionState.SUPERSEDED,
    }
)


class RuntimeRevision(BaseModel):
    """Complete materialized application runtime identity (§18.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_RUNTIME_REVISION_V1
    runtime_revision_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    application_release_id: str = _NON_EMPTY
    platform_version: str = _NON_EMPTY
    effective_roster_revision_id: str = _NON_EMPTY
    installed_agent_package_digests: tuple[str, ...] = ()
    materialized_runtime_lock_id: str | None = None
    materialized_runtime_lock_digest: str | None = None
    runtime_graph_digest: str | None = None
    materialization_artifact_digest: str | None = None
    materialization_topology: MaterializationTopology | None = None
    policy_certification_evidence_refs: tuple[AgentTrustEvidenceRef, ...] = ()
    revision_state: RuntimeRevisionState
    supersedes_revision_id: str | None = None
    rollback_target_revision_id: str | None = None
    activated_at: datetime | None = None

    @field_validator(
        "runtime_revision_id",
        "application_environment_id",
        "application_release_id",
        "platform_version",
        "effective_roster_revision_id",
        "materialized_runtime_lock_id",
        "materialized_runtime_lock_digest",
        "runtime_graph_digest",
        "materialization_artifact_digest",
        "supersedes_revision_id",
        "rollback_target_revision_id",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("installed_agent_package_digests")
    @classmethod
    def _strip_digests(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)

    @model_validator(mode="after")
    def _validate_state_requirements(self) -> RuntimeRevision:
        if self.revision_state in _ACTIVE_OR_VALIDATED:
            missing: list[str] = []
            if self.materialized_runtime_lock_id is None:
                missing.append("materialized_runtime_lock_id")
            if self.materialized_runtime_lock_digest is None:
                missing.append("materialized_runtime_lock_digest")
            if self.runtime_graph_digest is None:
                missing.append("runtime_graph_digest")
            if self.materialization_artifact_digest is None:
                missing.append("materialization_artifact_digest")
            if self.materialization_topology is None:
                missing.append("materialization_topology")
            if missing:
                raise ValueError(
                    f"{self.revision_state.value} runtime revision requires: "
                    + ", ".join(missing)
                )
        if self.revision_state is RuntimeRevisionState.ACTIVE and self.activated_at is None:
            raise ValueError("active runtime revision requires activated_at")
        return self
