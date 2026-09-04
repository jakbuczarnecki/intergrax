# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Federated agent discovery composing multiple discovery strategies (AC-4 Phase 4)."""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict, model_validator

from intergrax.agent_distribution.agent_discovery import (
    AgentDiscoveryCandidate,
    AgentDiscoveryContractError,
    AgentDiscoveryError,
    AgentDiscoveryIdentityConflict,
    AgentDiscoveryRequest,
    AgentDiscoveryResult,
    AgentDiscoveryStrategy,
    AgentDiscoveryStrategyId,
    discovery_candidate_sort_key,
)
from intergrax.agent_distribution.catalog import AgentDiscoveryCandidateIdentity

SCHEMA_FEDERATED_AGENT_DISCOVERY_RESULT_V1: Final = (
    "federated_agent_discovery_result.v1"
)
SCHEMA_FEDERATED_DISCOVERY_CANDIDATE_EVIDENCE_V1: Final = (
    "federated_discovery_candidate_evidence.v1"
)

FEDERATED_DISCOVERY_STRATEGY_ID: Final = AgentDiscoveryStrategyId(
    value="federated.v1",
)


def _strategy_id_sort_key(strategy_id: AgentDiscoveryStrategyId) -> str:
    return strategy_id.value


class FederatedDiscoveryError(AgentDiscoveryError):
    """Base error for federated discovery contract violations."""


class FederatedDiscoveryConfigurationError(
    FederatedDiscoveryError,
    AgentDiscoveryContractError,
):
    """Invalid federated discovery composition configuration."""


class FederatedDiscoveryChildResultError(
    FederatedDiscoveryError,
    AgentDiscoveryContractError,
):
    """Child strategy returned a result that violates federation contracts."""


class FederatedDiscoveryCandidateEvidence(BaseModel):
    """Typed multi-strategy provenance for one canonical candidate identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_FEDERATED_DISCOVERY_CANDIDATE_EVIDENCE_V1
    identity: AgentDiscoveryCandidateIdentity
    discovering_strategy_ids: tuple[AgentDiscoveryStrategyId, ...]

    @model_validator(mode="after")
    def _validate_strategy_ids(self) -> FederatedDiscoveryCandidateEvidence:
        if not self.discovering_strategy_ids:
            raise FederatedDiscoveryConfigurationError(
                "candidate evidence requires at least one discovering strategy id",
            )
        seen: set[str] = set()
        for strategy_id in self.discovering_strategy_ids:
            key = strategy_id.value
            if key in seen:
                raise FederatedDiscoveryConfigurationError(
                    "duplicate discovering strategy id in candidate evidence",
                )
            seen.add(key)
        return self


class FederatedAgentDiscoveryResult(AgentDiscoveryResult):
    """Federated discovery output with deterministic multi-strategy provenance."""

    schema_version: str = SCHEMA_FEDERATED_AGENT_DISCOVERY_RESULT_V1
    invoked_child_strategy_ids: tuple[AgentDiscoveryStrategyId, ...]
    candidate_evidence: tuple[FederatedDiscoveryCandidateEvidence, ...]

    @model_validator(mode="after")
    def _validate_evidence_alignment(self) -> FederatedAgentDiscoveryResult:
        if len(self.candidates) != len(self.candidate_evidence):
            raise FederatedDiscoveryConfigurationError(
                "candidate evidence must align one-to-one with candidates",
            )
        for candidate, evidence in zip(
            self.candidates,
            self.candidate_evidence,
            strict=True,
        ):
            if candidate.identity.sort_key != evidence.identity.sort_key:
                raise FederatedDiscoveryConfigurationError(
                    "candidate evidence identity must match candidate identity",
                )
        return self


def provenance_for(
    result: FederatedAgentDiscoveryResult,
    identity: AgentDiscoveryCandidateIdentity,
) -> FederatedDiscoveryCandidateEvidence | None:
    """Deterministic identity → federated evidence lookup."""
    target_key = identity.sort_key
    for evidence in result.candidate_evidence:
        if evidence.identity.sort_key == target_key:
            return evidence
    return None


def _validate_child_discovery_result(
    *,
    strategy: AgentDiscoveryStrategy,
    request: AgentDiscoveryRequest,
    result: AgentDiscoveryResult,
) -> None:
    if result.strategy_id != strategy.strategy_id:
        raise FederatedDiscoveryChildResultError(
            "child discovery result strategy_id does not match invoked strategy",
        )
    if result.request != request:
        raise FederatedDiscoveryChildResultError(
            "child discovery result request does not match federated request",
        )


def _validate_child_strategy_ids(
    strategies: tuple[AgentDiscoveryStrategy, ...],
) -> tuple[AgentDiscoveryStrategy, ...]:
    if not strategies:
        raise FederatedDiscoveryConfigurationError(
            "federated discovery requires at least one child strategy",
        )
    seen: set[str] = set()
    for strategy in strategies:
        key = strategy.strategy_id.value
        if key in seen:
            raise FederatedDiscoveryConfigurationError(
                "duplicate child strategy_id in federated discovery configuration",
            )
        seen.add(key)
    return strategies


def merge_federated_candidates(
    observations: tuple[tuple[AgentDiscoveryStrategyId, AgentDiscoveryCandidate], ...],
) -> tuple[
    tuple[AgentDiscoveryCandidate, ...],
    tuple[FederatedDiscoveryCandidateEvidence, ...],
]:
    """Union child observations with exact-duplicate dedupe and conflict fail-closed."""
    grouped: dict[
        tuple[str, str, str, str, str],
        tuple[AgentDiscoveryCandidate, list[AgentDiscoveryStrategyId]],
    ] = {}
    for strategy_id, candidate in observations:
        identity_key = candidate.identity.sort_key
        existing = grouped.get(identity_key)
        if existing is None:
            grouped[identity_key] = (candidate, [strategy_id])
            continue
        canonical_candidate, strategy_ids = existing
        if canonical_candidate != candidate:
            raise AgentDiscoveryIdentityConflict(
                "conflicting discovery facts for the same canonical candidate identity",
            )
        strategy_ids.append(strategy_id)

    ordered_keys = sorted(grouped.keys())
    candidates: list[AgentDiscoveryCandidate] = []
    evidence: list[FederatedDiscoveryCandidateEvidence] = []
    for identity_key in ordered_keys:
        candidate, strategy_ids = grouped[identity_key]
        candidates.append(candidate)
        evidence.append(
            FederatedDiscoveryCandidateEvidence(
                identity=candidate.identity,
                discovering_strategy_ids=tuple(
                    sorted(strategy_ids, key=_strategy_id_sort_key),
                ),
            ),
        )
    return tuple(candidates), tuple(evidence)


class FederatedAgentDiscoveryStrategy:
    """Compose multiple discovery strategies into one deterministic federated result."""

    def __init__(self, strategies: tuple[AgentDiscoveryStrategy, ...]) -> None:
        self._strategies = _validate_child_strategy_ids(strategies)

    @property
    def strategy_id(self) -> AgentDiscoveryStrategyId:
        return FEDERATED_DISCOVERY_STRATEGY_ID

    def discover(self, request: AgentDiscoveryRequest) -> FederatedAgentDiscoveryResult:
        observations: list[
            tuple[AgentDiscoveryStrategyId, AgentDiscoveryCandidate]
        ] = []
        for strategy in self._strategies:
            child_result = strategy.discover(request)
            _validate_child_discovery_result(
                strategy=strategy,
                request=request,
                result=child_result,
            )
            for candidate in child_result.candidates:
                observations.append((strategy.strategy_id, candidate))

        candidates, evidence = merge_federated_candidates(tuple(observations))
        invoked_child_strategy_ids = tuple(
            sorted(
                (strategy.strategy_id for strategy in self._strategies),
                key=_strategy_id_sort_key,
            ),
        )
        return FederatedAgentDiscoveryResult(
            strategy_id=FEDERATED_DISCOVERY_STRATEGY_ID,
            request=request,
            candidates=tuple(
                sorted(candidates, key=discovery_candidate_sort_key),
            ),
            invoked_child_strategy_ids=invoked_child_strategy_ids,
            candidate_evidence=evidence,
        )
