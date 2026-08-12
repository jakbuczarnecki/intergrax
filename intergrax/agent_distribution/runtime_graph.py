# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Candidate runtime graph reference contracts (AGENT_DISTRIBUTION §17)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator

from intergrax.agent_distribution._digest import normalize_package_digest

_NON_EMPTY = Field(min_length=1)

SCHEMA_CANDIDATE_APPLICATION_RUNTIME_GRAPH_V1: Final = (
    "candidate_application_runtime_graph.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class RuntimeGraphAgentRef(BaseModel):
    """Reference to one agent node in the candidate runtime graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    logical_agent_id: str = _NON_EMPTY
    distribution_package_id: str = _NON_EMPTY
    package_digest: str = _NON_EMPTY

    @field_validator("logical_agent_id", "distribution_package_id")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)


class RuntimeGraphThirdPartyRef(BaseModel):
    """App-declared third-party distribution reference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    distribution_name: str = _NON_EMPTY
    version: str = _NON_EMPTY

    @field_validator("distribution_name", "version")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class RuntimeGraphTierViolation(BaseModel):
    """Tier violation evidence — non-empty blocks activation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str = _NON_EMPTY
    detail: str = _NON_EMPTY

    @field_validator("code", "detail")
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)


class CandidateApplicationRuntimeGraph(BaseModel):
    """Structural runtime graph reference owned by distribution (§17)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CANDIDATE_APPLICATION_RUNTIME_GRAPH_V1
    graph_schema_version: str = _NON_EMPTY
    application_id: str = _NON_EMPTY
    runtime_graph_digest: str = _NON_EMPTY
    materialized_runtime_lock_id: str = _NON_EMPTY
    direct_agents: tuple[RuntimeGraphAgentRef, ...]
    transitive_agents: tuple[RuntimeGraphAgentRef, ...] = ()
    direct_third_party_distributions: tuple[RuntimeGraphThirdPartyRef, ...] = ()
    tier_violations: tuple[RuntimeGraphTierViolation, ...] = ()

    @field_validator(
        "graph_schema_version",
        "application_id",
        "runtime_graph_digest",
        "materialized_runtime_lock_id",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)
