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


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class MaterializationInput(BaseModel):
    """Logical materialization input — topology-agnostic (§19.1)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_MATERIALIZATION_INPUT_V1
    runtime_revision: RuntimeRevision
    materialized_runtime_lock: MaterializedRuntimeLock
    candidate_runtime_graph: CandidateApplicationRuntimeGraph
    effective_roster: EffectiveRoster
    application_build_context_ref: str = _NON_EMPTY

    @field_validator("application_build_context_ref")
    @classmethod
    def _strip_ref(cls, value: str) -> str:
        return _strip_required(value)


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
