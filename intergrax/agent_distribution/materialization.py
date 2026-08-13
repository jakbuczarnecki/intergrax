# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Topology-neutral materialization contracts (AGENT_DISTRIBUTION §19)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution.dependency import MaterializedRuntimeLock
from intergrax.agent_distribution.roster import EffectiveRoster
from intergrax.agent_distribution.runtime_graph import CandidateApplicationRuntimeGraph
from intergrax.agent_distribution.runtime_revision import MaterializationTopology, RuntimeRevision

_NON_EMPTY = Field(min_length=1)

SCHEMA_MATERIALIZATION_INPUT_V1: Final = "materialization_input.v1"
SCHEMA_MATERIALIZATION_OUTPUT_V1: Final = "materialization_output.v1"
SCHEMA_APPLICATION_BUILD_CONTEXT_V1: Final = "application_build_context.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class ApplicationBuildContext(BaseModel):
    """Neutral Tier-0 physical-build inputs — not application business logic (§19.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_APPLICATION_BUILD_CONTEXT_V1
    application_id: str = _NON_EMPTY
    application_release_id: str = _NON_EMPTY
    application_environment_id: str = _NON_EMPTY
    source_context_root: str = _NON_EMPTY
    platform_version: str = _NON_EMPTY
    python_version: str = _NON_EMPTY
    output_root: str = _NON_EMPTY
    application_source_root: str = _NON_EMPTY
    agent_source_roots: tuple[tuple[str, str], ...] = ()
    health_check_path: str = "/health"
    docker_image_tag: str | None = None
    entrypoint_module: str | None = None

    @field_validator(
        "application_id",
        "application_release_id",
        "application_environment_id",
        "source_context_root",
        "platform_version",
        "python_version",
        "output_root",
        "application_source_root",
        "health_check_path",
        "docker_image_tag",
        "entrypoint_module",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @field_validator("agent_source_roots")
    @classmethod
    def _validate_agent_source_roots(
        cls,
        value: tuple[tuple[str, str], ...],
    ) -> tuple[tuple[str, str], ...]:
        normalized: list[tuple[str, str]] = []
        for package_id, rel_path in value:
            normalized.append((_strip_required(package_id), _strip_required(rel_path)))
        return tuple(normalized)


class MaterializationInput(BaseModel):
    """Logical materialization input — topology-agnostic (§19.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MATERIALIZATION_INPUT_V1
    runtime_revision: RuntimeRevision
    materialized_runtime_lock: MaterializedRuntimeLock
    candidate_runtime_graph: CandidateApplicationRuntimeGraph
    effective_roster: EffectiveRoster
    application_build_context: ApplicationBuildContext

    @field_validator("application_build_context")
    @classmethod
    def _validate_build_context(
        cls,
        value: ApplicationBuildContext,
    ) -> ApplicationBuildContext:
        return value


class MaterializationOutput(BaseModel):
    """Logical materialization output — topology-agnostic (§19.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MATERIALIZATION_OUTPUT_V1
    materialization_artifact_digest: str = _NON_EMPTY
    artifact_locator: str = _NON_EMPTY
    health_check_evidence_ref: str | None = None
    runtime_graph_manifest_path: str = _NON_EMPTY
    topology: MaterializationTopology

    @field_validator(
        "materialization_artifact_digest",
        "artifact_locator",
        "health_check_evidence_ref",
        "runtime_graph_manifest_path",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)
