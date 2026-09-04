# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Canonical capability identity, requirement, and deterministic matching (AC-4 Phase 1)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution.errors import AgentDistributionError

_NON_EMPTY = Field(min_length=1)

SCHEMA_CAPABILITY_ID_V1: Final = "capability_id.v1"
SCHEMA_AGENT_CAPABILITY_DECLARATION_V1: Final = "agent_capability_declaration.v1"
SCHEMA_CAPABILITY_REQUIREMENT_V1: Final = "capability_requirement.v1"
SCHEMA_AGENT_CAPABILITY_REQUIREMENT_V1: Final = "agent_capability_requirement.v1"
SCHEMA_AGENT_CAPABILITY_CANDIDATE_V1: Final = "agent_capability_candidate.v1"
SCHEMA_CAPABILITY_MATCH_RESULT_V1: Final = "capability_match_result.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class CapabilityModelError(AgentDistributionError):
    """Malformed capability authority model."""


class CapabilityRequirementError(CapabilityModelError):
    """Invalid capability requirement specification."""


class CapabilityId(BaseModel):
    """Stable, normalized capability identity for deterministic matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_ID_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class AgentCapabilityDeclaration(BaseModel):
    """One declared capability on an agent/package candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_CAPABILITY_DECLARATION_V1
    capability_id: CapabilityId


class CapabilityRequirement(BaseModel):
    """Single required or optional capability in an application requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_REQUIREMENT_V1
    capability_id: CapabilityId
    required: bool = True


class AgentCapabilityRequirement(BaseModel):
    """Aggregate capability requirement — at least one entry, no duplicate ids."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_CAPABILITY_REQUIREMENT_V1
    requirements: tuple[CapabilityRequirement, ...]

    @model_validator(mode="after")
    def _validate_requirements(self) -> AgentCapabilityRequirement:
        if not self.requirements:
            raise CapabilityRequirementError(
                "agent capability requirement must contain at least one capability",
            )
        seen: set[str] = set()
        for item in self.requirements:
            key = item.capability_id.value
            if key in seen:
                raise CapabilityRequirementError(
                    f"duplicate capability requirement for capability_id={key!r}",
                )
            seen.add(key)
        return self

    @property
    def required_capability_ids(self) -> frozenset[CapabilityId]:
        return frozenset(
            item.capability_id for item in self.requirements if item.required
        )

    @property
    def optional_capability_ids(self) -> frozenset[CapabilityId]:
        return frozenset(
            item.capability_id for item in self.requirements if not item.required
        )


class AgentCapabilityCandidate(BaseModel):
    """Capability-bearing match input — source-agnostic, pre-installation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_CAPABILITY_CANDIDATE_V1
    candidate_id: str = _NON_EMPTY
    capabilities: tuple[AgentCapabilityDeclaration, ...] = ()
    source_ref: str | None = None

    @field_validator("candidate_id", "source_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @model_validator(mode="after")
    def _validate_capabilities(self) -> AgentCapabilityCandidate:
        seen: set[str] = set()
        for declaration in self.capabilities:
            key = declaration.capability_id.value
            if key in seen:
                raise CapabilityModelError(
                    f"duplicate capability declaration for capability_id={key!r}",
                )
            seen.add(key)
        return self

    @property
    def declared_capability_ids(self) -> frozenset[CapabilityId]:
        return frozenset(declaration.capability_id for declaration in self.capabilities)


class CapabilityMatchResult(BaseModel):
    """Auditable capability match evidence for one candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_CAPABILITY_MATCH_RESULT_V1
    candidate_id: str = _NON_EMPTY
    eligible: bool
    matched_required: tuple[CapabilityId, ...]
    missing_required: tuple[CapabilityId, ...]
    matched_optional: tuple[CapabilityId, ...]
    unsupported_constraints: tuple[str, ...] = ()

    @field_validator("candidate_id")
    @classmethod
    def _strip_candidate_id(cls, value: str) -> str:
        return _strip_required(value)


def _sorted_capability_ids(values: frozenset[CapabilityId]) -> tuple[CapabilityId, ...]:
    return tuple(sorted(values, key=lambda item: item.value))


class CapabilityMatcher:
    """Pure deterministic capability matcher — no catalog, install, or trust I/O."""

    def match(
        self,
        *,
        requirement: AgentCapabilityRequirement,
        candidate: AgentCapabilityCandidate,
    ) -> CapabilityMatchResult:
        declared = candidate.declared_capability_ids
        required = requirement.required_capability_ids
        optional = requirement.optional_capability_ids

        matched_required = required & declared
        missing_required = required - declared
        matched_optional = optional & declared
        eligible = not missing_required

        return CapabilityMatchResult(
            candidate_id=candidate.candidate_id,
            eligible=eligible,
            matched_required=_sorted_capability_ids(matched_required),
            missing_required=_sorted_capability_ids(missing_required),
            matched_optional=_sorted_capability_ids(matched_optional),
        )

    def find_matches(
        self,
        *,
        requirement: AgentCapabilityRequirement,
        candidates: Sequence[AgentCapabilityCandidate],
    ) -> tuple[CapabilityMatchResult, ...]:
        results = (
            self.match(requirement=requirement, candidate=candidate)
            for candidate in candidates
        )
        return tuple(sorted(results, key=lambda item: item.candidate_id))

    def find_eligible(
        self,
        *,
        requirement: AgentCapabilityRequirement,
        candidates: Sequence[AgentCapabilityCandidate],
    ) -> tuple[CapabilityMatchResult, ...]:
        return tuple(
            result
            for result in self.find_matches(
                requirement=requirement, candidates=candidates
            )
            if result.eligible
        )


def build_agent_capability_requirement(
    *,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> AgentCapabilityRequirement:
    """Construct a validated requirement from normalized capability id strings."""
    requirements: list[CapabilityRequirement] = [
        CapabilityRequirement(capability_id=CapabilityId(value=item), required=True)
        for item in required
    ]
    requirements.extend(
        CapabilityRequirement(capability_id=CapabilityId(value=item), required=False)
        for item in optional
    )
    return AgentCapabilityRequirement(requirements=tuple(requirements))


def build_agent_capability_candidate(
    *,
    candidate_id: str,
    capability_ids: Sequence[str] = (),
    source_ref: str | None = None,
) -> AgentCapabilityCandidate:
    """Construct a validated candidate from normalized capability id strings."""
    return AgentCapabilityCandidate(
        candidate_id=candidate_id,
        capabilities=tuple(
            AgentCapabilityDeclaration(capability_id=CapabilityId(value=item))
            for item in capability_ids
        ),
        source_ref=source_ref,
    )
