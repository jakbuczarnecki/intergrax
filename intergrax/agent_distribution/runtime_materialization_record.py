# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical runtime materialization authority record (ADR-AGENT-006)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.errors import RuntimeMaterializationConflict
from intergrax.agent_distribution.runtime_revision import (
    MaterializationTopology,
    RuntimeRevision,
)

_NON_EMPTY = Field(min_length=1)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RuntimeMaterializationRecord(BaseModel):
    """Immutable canonical lifecycle record for one materialized runtime revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_revision_id: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    materialization_topology: MaterializationTopology
    artifact_locator: str = _NON_EMPTY
    materialization_artifact_digest: str = _NON_EMPTY
    materialized_runtime_lock_id: str = _NON_EMPTY
    materialized_runtime_lock_digest: str = _NON_EMPTY

    @field_validator(
        "runtime_revision_id",
        "application_id",
        "application_environment_id",
        "artifact_locator",
        "materialized_runtime_lock_id",
    )
    @classmethod
    def _strip_required_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator(
        "materialization_artifact_digest",
        "materialized_runtime_lock_digest",
    )
    @classmethod
    def _validate_digests(cls, value: str) -> str:
        return normalize_package_digest(value)


def validate_runtime_materialization_record_matches_revision(
    revision: RuntimeRevision,
    record: RuntimeMaterializationRecord,
) -> None:
    """Fail closed when canonical materialization authority diverges from revision."""
    if record.runtime_revision_id != revision.runtime_revision_id:
        raise RuntimeMaterializationConflict(
            "runtime materialization revision id mismatch"
        )
    if record.application_id != revision.application_id:
        raise RuntimeMaterializationConflict(
            "runtime materialization application id mismatch"
        )
    if record.application_environment_id != revision.application_environment_id:
        raise RuntimeMaterializationConflict(
            "runtime materialization application environment id mismatch"
        )
    if revision.materialization_topology is not None:
        if record.materialization_topology != revision.materialization_topology:
            raise RuntimeMaterializationConflict(
                "runtime materialization topology mismatch"
            )
    if revision.materialization_artifact_digest is not None:
        if (
            record.materialization_artifact_digest
            != revision.materialization_artifact_digest
        ):
            raise RuntimeMaterializationConflict(
                "runtime materialization artifact digest mismatch"
            )
    if revision.materialized_runtime_lock_id is not None:
        if record.materialized_runtime_lock_id != revision.materialized_runtime_lock_id:
            raise RuntimeMaterializationConflict(
                "runtime materialization lock id mismatch"
            )
    if revision.materialized_runtime_lock_digest is not None:
        if (
            record.materialized_runtime_lock_digest
            != revision.materialized_runtime_lock_digest
        ):
            raise RuntimeMaterializationConflict(
                "runtime materialization lock digest mismatch"
            )
