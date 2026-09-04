# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pluggable agent selection contracts above capability matching (AC-4 Phase 3)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution.capability_matching import (
    AgentCapabilityRequirement,
    CapabilityMatchResult,
)
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity
from intergrax.agent_distribution.errors import AgentDistributionError

_NON_EMPTY = Field(min_length=1)

SCHEMA_AGENT_SELECTION_STRATEGY_ID_V1: Final = "agent_selection_strategy_id.v1"
SCHEMA_AGENT_SELECTION_CONTEXT_V1: Final = "agent_selection_context.v1"
SCHEMA_AGENT_SELECTION_REQUEST_V1: Final = "agent_selection_request.v1"
SCHEMA_AGENT_SELECTION_DECISION_V1: Final = "agent_selection_decision.v1"

DETERMINISTIC_IDENTITY_STRATEGY_ID: Final = "deterministic.identity"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class AgentSelectionError(AgentDistributionError):
    """Base error for agent selection contract violations."""


class AgentSelectionContractError(AgentSelectionError):
    """Malformed selection request or decision."""


class AgentSelectionNoEligibleCandidate(AgentSelectionError):
    """Selection required a candidate but none were eligible."""


class AgentSelectionIdentityConflict(AgentSelectionContractError):
    """Duplicate canonical candidate identities in one selection request."""


class SelectionOutcome(StrEnum):
    """Whether a candidate was selected or none were eligible."""

    SELECTED = "selected"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"


class SelectionDecisionBasis(StrEnum):
    """Typed, auditable basis for a selection decision — not free-form prose."""

    STABLE_IDENTITY_ORDER = "stable_identity_order"


class AgentSelectionStrategyId(BaseModel):
    """Stable plugin identifier — not derived from class name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_SELECTION_STRATEGY_ID_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class AgentSelectionContext(BaseModel):
    """Minimal versioned selection context — extensible without metadata bags."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_SELECTION_CONTEXT_V1


class AgentSelectionRequest(BaseModel):
    """Eligible capability matches ready for strategy evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_SELECTION_REQUEST_V1
    requirement: AgentCapabilityRequirement
    eligible_matches: tuple[CapabilityMatchResult, ...] = ()
    selection_context: AgentSelectionContext = AgentSelectionContext()

    @model_validator(mode="after")
    def _validate_eligible_matches(self) -> AgentSelectionRequest:
        seen: dict[tuple[str, str, str, str, str], CapabilityMatchResult] = {}
        for match in self.eligible_matches:
            if not match.eligible:
                raise AgentSelectionContractError(
                    "selection request must contain only eligible capability matches",
                )
            key = match.identity.sort_key
            existing = seen.get(key)
            if existing is not None:
                raise AgentSelectionIdentityConflict(
                    "duplicate canonical candidate identity in selection request",
                )
            seen[key] = match
        return self


class AgentSelectionDecision(BaseModel):
    """Auditable selection outcome — explains why one candidate was chosen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_AGENT_SELECTION_DECISION_V1
    strategy_id: AgentSelectionStrategyId
    outcome: SelectionOutcome
    selected_identity: AgentDiscoveryCandidateIdentity | None = None
    considered_candidates: tuple[AgentDiscoveryCandidateIdentity, ...] = ()
    decision_basis: SelectionDecisionBasis | None = None

    @model_validator(mode="after")
    def _validate_outcome(self) -> AgentSelectionDecision:
        if self.outcome is SelectionOutcome.SELECTED:
            if self.selected_identity is None:
                raise AgentSelectionContractError(
                    "selected outcome requires selected_identity",
                )
            if self.decision_basis is None:
                raise AgentSelectionContractError(
                    "selected outcome requires decision_basis",
                )
            return self
        if self.selected_identity is not None:
            raise AgentSelectionContractError(
                "no_eligible_candidate outcome must not include selected_identity",
            )
        if self.decision_basis is not None:
            raise AgentSelectionContractError(
                "no_eligible_candidate outcome must not include decision_basis",
            )
        return self


def match_identity_sort_key(
    match: CapabilityMatchResult,
) -> tuple[str, str, str, str, str]:
    return match.identity.sort_key


def sorted_eligible_identities(
    matches: tuple[CapabilityMatchResult, ...],
) -> tuple[AgentDiscoveryCandidateIdentity, ...]:
    """Canonical ordering of eligible identities for audit evidence."""
    return tuple(
        match.identity for match in sorted(matches, key=match_identity_sort_key)
    )


def build_agent_selection_request(
    *,
    requirement: AgentCapabilityRequirement,
    eligible_matches: tuple[CapabilityMatchResult, ...],
    selection_context: AgentSelectionContext | None = None,
) -> AgentSelectionRequest:
    """Construct a validated selection request from matcher-eligible results."""
    return AgentSelectionRequest(
        requirement=requirement,
        eligible_matches=eligible_matches,
        selection_context=selection_context or AgentSelectionContext(),
    )


def require_selected_identity(
    decision: AgentSelectionDecision,
) -> AgentDiscoveryCandidateIdentity:
    """Fail-closed accessor when orchestration requires a selected candidate."""
    if decision.outcome is not SelectionOutcome.SELECTED:
        raise AgentSelectionNoEligibleCandidate(
            "no eligible candidate was selected",
        )
    if decision.selected_identity is None:
        raise AgentSelectionNoEligibleCandidate(
            "selection decision missing selected_identity",
        )
    return decision.selected_identity


class AgentSelectionStrategy(Protocol):
    """Structural selection plugin — no registry or service locator."""

    @property
    def strategy_id(self) -> AgentSelectionStrategyId:
        """Stable strategy identifier."""

    def select(self, request: AgentSelectionRequest) -> AgentSelectionDecision:
        """Choose one eligible candidate without discovery, matching, or lifecycle I/O."""


class DeterministicIdentitySelectionStrategy:
    """Deterministic baseline selector — stable identity order, not semantically best."""

    @property
    def strategy_id(self) -> AgentSelectionStrategyId:
        return AgentSelectionStrategyId(value=DETERMINISTIC_IDENTITY_STRATEGY_ID)

    def select(self, request: AgentSelectionRequest) -> AgentSelectionDecision:
        considered = sorted_eligible_identities(request.eligible_matches)
        if not considered:
            return AgentSelectionDecision(
                strategy_id=self.strategy_id,
                outcome=SelectionOutcome.NO_ELIGIBLE_CANDIDATE,
                considered_candidates=(),
            )
        selected = considered[0]
        return AgentSelectionDecision(
            strategy_id=self.strategy_id,
            outcome=SelectionOutcome.SELECTED,
            selected_identity=selected,
            considered_candidates=considered,
            decision_basis=SelectionDecisionBasis.STABLE_IDENTITY_ORDER,
        )
