# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical runtime materialization authority record (ADR-AGENT-006)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.runtime_revision import MaterializationTopology

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
