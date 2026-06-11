# © Artur Czarnecki. All rights reserved.

"""Capability-token agent resolution for Nexus (ACP-CON-6)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.agents.agent_contract import Agent
from intergrax.contracts.task_routing import validate_task_routing_payload
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.task.task import Task


@dataclass(frozen=True, slots=True)
class CapabilityRouteResult:
    """Outcome of capability-based agent resolution."""

    capability: str
    candidates: tuple[Agent, ...]
    selected: Agent | None
    selection_reason: str


def validate_task_for_capability_routing(task: Task) -> None:
    """Ensure task payload uses capability routing contract."""
    validate_task_routing_payload(
        metadata=task.metadata,
        context_metadata=task.context.metadata,
    )


def resolve_agents_for_capability(
    registry: AgentRegistry,
    capability: str,
    *,
    production_mode: bool = False,
) -> list[Agent]:
    """Resolve registry agents by capability token (§37.6)."""
    token = capability.strip()
    if not token:
        return []
    return registry.find_by_capability(token, production_mode=production_mode)


def select_best_capability_match(
    registry: AgentRegistry,
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
    if not candidates:
        return CapabilityRouteResult(
            capability=capability,
            candidates=(),
            selected=None,
            selection_reason="no_capability_match",
        )

    best: tuple[float, Agent] | None = None
    for agent in candidates:
        result = agent.can_handle(task.context)
        if not result.matched:
            continue
        if best is None or result.score > best[0]:
            best = (result.score, agent)

    if best is None:
        return CapabilityRouteResult(
            capability=capability,
            candidates=tuple(candidates),
            selected=candidates[0],
            selection_reason="capability_first_match",
        )
    return CapabilityRouteResult(
        capability=capability,
        candidates=tuple(candidates),
        selected=best[1],
        selection_reason="capability_best_score",
    )
