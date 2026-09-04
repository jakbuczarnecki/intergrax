# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable agent discovery contracts above catalog/providers (AC-4 Phase 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from intergrax.agent_distribution.capability_matching import (
        AgentCapabilityCandidate,
    )

from intergrax.agent_distribution.agent_project_metadata import (
    AgentPackageContractDeclaration,
)
from intergrax.agent_distribution.capability_matching import (
    AgentCapabilityDeclaration,
    AgentCapabilityRequirement,
    CapabilityId,
    CapabilityModelError,
)
from intergrax.agent_distribution.catalog import (
    AgentDiscoveryCandidateIdentity,
)
from intergrax.agent_distribution.errors import AgentDistributionError

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_DISCOVERY_STRATEGY_ID_V1: Final = "agent_discovery_strategy_id.v1"
SCHEMA_AGENT_DISCOVERY_CANDIDATE_V1: Final = "agent_discovery_candidate.v1"
SCHEMA_AGENT_DISCOVERY_REQUEST_V1: Final = "agent_discovery_request.v1"
SCHEMA_AGENT_DISCOVERY_RESULT_V1: Final = "agent_discovery_result.v1"
SCHEMA_AGENT_DISCOVERY_SCOPE_V1: Final = "agent_discovery_scope.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentDiscoveryError(AgentDistributionError):
    """Base error for agent discovery contract violations."""


class AgentDiscoveryContractError(AgentDiscoveryError):
    """Malformed discovery request, result, or candidate projection."""


class AgentDiscoveryIdentityConflict(AgentDiscoveryContractError):
    """Duplicate or conflicting canonical candidate identities in one result."""


class AgentDiscoveryStrategyId(BaseModel):
    """Stable plugin identifier — not derived from class name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_DISCOVERY_STRATEGY_ID_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class AgentDiscoveryScope(BaseModel):
    """Optional discovery boundary — origin-agnostic, not user-specific."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_DISCOVERY_SCOPE_V1
    allowed_catalog_source_ids: tuple[str, ...] = ()

    @field_validator("allowed_catalog_source_ids")
    @classmethod
    def _strip_source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(_strip_required(item) for item in value)


class AgentDiscoveryRequest(BaseModel):
    """What to discover — structured capability need, not provider query semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_DISCOVERY_REQUEST_V1
    requirement: AgentCapabilityRequirement
    scope: AgentDiscoveryScope = AgentDiscoveryScope()


class AgentDiscoveryCandidate(BaseModel):
    """Discovered capability-bearing candidate with source-qualified identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_DISCOVERY_CANDIDATE_V1
    identity: AgentDiscoveryCandidateIdentity
    capabilities: tuple[AgentCapabilityDeclaration, ...] = ()
    catalog_entry_id: str | None = None
    artifact_locator: str | None = None

    @field_validator("catalog_entry_id", "artifact_locator")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _strip_required(value)

    @model_validator(mode="after")
    def _validate_capabilities(self) -> AgentDiscoveryCandidate:
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


class AgentDiscoveryResult(BaseModel):
    """Deterministic discovery output — zero candidates is valid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_DISCOVERY_RESULT_V1
    strategy_id: AgentDiscoveryStrategyId
    request: AgentDiscoveryRequest
    candidates: tuple[AgentDiscoveryCandidate, ...] = ()


def discovery_candidate_sort_key(
    candidate: AgentDiscoveryCandidate,
) -> tuple[str, str, str, str, str]:
    return candidate.identity.sort_key


def normalize_discovery_candidates(
    candidates: tuple[AgentDiscoveryCandidate, ...],
) -> tuple[AgentDiscoveryCandidate, ...]:
    """Deterministic ordering with fail-closed duplicate and conflict detection."""
    ordered = tuple(sorted(candidates, key=discovery_candidate_sort_key))
    seen: dict[tuple[str, str, str, str, str], AgentDiscoveryCandidate] = {}
    for candidate in ordered:
        key = candidate.identity.sort_key
        existing = seen.get(key)
        if existing is None:
            seen[key] = candidate
            continue
        if existing == candidate:
            raise AgentDiscoveryIdentityConflict(
                "duplicate canonical discovery candidate identity in one result",
            )
        raise AgentDiscoveryIdentityConflict(
            "conflicting discovery facts for the same canonical candidate identity",
        )
    return ordered


def build_agent_discovery_result(
    *,
    strategy_id: AgentDiscoveryStrategyId,
    request: AgentDiscoveryRequest,
    candidates: tuple[AgentDiscoveryCandidate, ...],
) -> AgentDiscoveryResult:
    return AgentDiscoveryResult(
        strategy_id=strategy_id,
        request=request,
        candidates=normalize_discovery_candidates(candidates),
    )


def project_package_contract_capabilities(
    contract: AgentPackageContractDeclaration,
) -> tuple[AgentCapabilityDeclaration, ...]:
    """Project package contract capability strings into canonical declarations."""
    return tuple(
        AgentCapabilityDeclaration(capability_id=CapabilityId(value=item))
        for item in contract.capabilities
    )


def project_to_capability_candidate(
    candidate: AgentDiscoveryCandidate,
) -> AgentCapabilityCandidate:
    """Lossless matcher projection preserving canonical candidate identity."""
    from intergrax.agent_distribution.capability_matching import (
        AgentCapabilityCandidate,
    )

    return AgentCapabilityCandidate(
        identity=candidate.identity,
        capabilities=candidate.capabilities,
    )


class AgentDiscoveryStrategy(Protocol):
    """Structural discovery plugin — no registry or service locator."""

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        """Stable strategy identifier."""

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        """Return source-qualified candidates without matching or lifecycle I/O."""


class StaticAgentDiscoveryStrategy:
    """Reference strategy returning a fixed candidate set for tests and composition."""

    def __init__(
        self,
        *,
        strategy_id: AgentDiscoveryStrategyId,
        candidates: tuple[AgentDiscoveryCandidate, ...],
    ) -> None:
        self._strategy_id = strategy_id
        self._candidates = candidates

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return self._strategy_id

    def discover(self, request: AgentDiscoveryRequest) -> AgentDiscoveryResult:
        allowed = request.scope.allowed_catalog_source_ids
        if allowed:
            allowed_set = frozenset(allowed)
            filtered = tuple(
                candidate
                for candidate in self._candidates
                if candidate.identity.source.catalog_source_id in allowed_set
            )
        else:
            filtered = self._candidates
        return build_agent_discovery_result(
            strategy_id=self._strategy_id,
            request=request,
            candidates=filtered,
        )
