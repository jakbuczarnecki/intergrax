# © Artur Czarnecki. All rights reserved.

"""Capability-token agent resolution for Nexus (ACP-CON-6)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.contracts.capability import CapabilityMatchResult
from intergrax.contracts.routable_tier2_agent import (
    AgentRoutingContractError,
    RoutableTier2Agent,
)
from intergrax.contracts.task_envelope import TaskEnvelope
from intergrax.contracts.task_routing import validate_task_routing_payload
from intergrax.contracts.tier2_agent import Tier2Agent
from intergrax.runtime.registry.agent_registry_read import AgentRegistryRead
from intergrax.runtime.task.agent_capability_intake import task_envelope_for_agent_capability_match
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class RoutableAgentMatchEvidence:
    """Single can_handle evaluation for one routable candidate."""

    agent: RoutableTier2Agent
    match: CapabilityMatchResult


@dataclass(frozen=True, slots=True)
class CapabilityRouteResult:
    """Outcome of capability-based agent resolution."""

    capability: str
    candidates: tuple[Tier2Agent, ...]
    selected: RoutableTier2Agent | None
    selection_reason: str
    selected_match: CapabilityMatchResult | None = None


def evaluate_routable_candidates(
    envelope: TaskEnvelope,
    candidates: Sequence[Tier2Agent],
) -> tuple[RoutableAgentMatchEvidence, ...]:
    """Canonical can_handle scoring — at most one call per routable candidate."""
    evidence: list[RoutableAgentMatchEvidence] = []
    for agent in candidates:
        if not isinstance(agent, RoutableTier2Agent):
            continue
        evidence.append(
            RoutableAgentMatchEvidence(agent=agent, match=agent.can_handle(envelope))
        )
    return tuple(evidence)


def _best_matched_evidence(
    evidence: Sequence[RoutableAgentMatchEvidence],
) -> RoutableAgentMatchEvidence | None:
    best: RoutableAgentMatchEvidence | None = None
    for item in evidence:
        if not item.match.matched:
            continue
        if best is None or item.match.score > best.match.score:
            best = item
    return best


def select_best_matched_routable_agent(
    *,
    envelope: TaskEnvelope,
    candidates: Sequence[Tier2Agent],
) -> RoutableTier2Agent | None:
    """Highest-scoring matched routable agent; no fallback when nothing matched."""
    best = _best_matched_evidence(evaluate_routable_candidates(envelope, candidates))
    return best.agent if best is not None else None


def validate_task_for_capability_routing(task: Task) -> None:
    """Ensure task payload uses capability routing contract."""
    validate_task_routing_payload(
        metadata=task.metadata,
        context_metadata=task.context.metadata,
    )


def resolve_agents_for_capability(
    registry: AgentRegistryRead,
    capability: str,
    *,
    production_mode: bool = False,
) -> list[Tier2Agent]:
    """Resolve registry agents by capability token (§37.6)."""
    token = capability.strip()
    if not token:
        return []
    return registry.find_by_capability(token, production_mode=production_mode)


def select_best_routable_agent(
    *,
    capability: str,
    envelope: TaskEnvelope,
    candidates: Sequence[Tier2Agent],
) -> CapabilityRouteResult:
    """Pick highest-scoring RoutableTier2Agent among pre-resolved capability matches."""
    if not candidates:
        return CapabilityRouteResult(
            capability=capability,
            candidates=(),
            selected=None,
            selection_reason="no_capability_match",
        )

    candidate_tuple = tuple(candidates)
    scored = evaluate_routable_candidates(envelope, candidate_tuple)
    routable_candidates = [item.agent for item in scored]
    best = _best_matched_evidence(scored)

    if best is None:
        if not routable_candidates:
            raise AgentRoutingContractError(
                f"No registered agent for capability '{capability}' implements "
                "RoutableTier2Agent (can_handle)"
            )
        fallback_evidence = scored[0]
        return CapabilityRouteResult(
            capability=capability,
            candidates=candidate_tuple,
            selected=fallback_evidence.agent,
            selection_reason="capability_first_match",
            selected_match=(
                fallback_evidence.match
                if fallback_evidence.match.matched
                else None
            ),
        )
    return CapabilityRouteResult(
        capability=capability,
        candidates=candidate_tuple,
        selected=best.agent,
        selection_reason="capability_best_score",
        selected_match=best.match,
    )


def select_best_capability_match(
    registry: AgentRegistryRead,
    task: Task,
    capability: str,
    *,
    production_mode: bool = False,
) -> CapabilityRouteResult:
    """Pick highest-scoring agent among capability matches."""
    validate_task_for_capability_routing(task)
    candidates = resolve_agents_for_capability(
        registry,
        capability,
        production_mode=production_mode,
    )
    envelope = task_envelope_for_agent_capability_match(task)
    return select_best_routable_agent(
        capability=capability,
        envelope=envelope,
        candidates=candidates,
    )
