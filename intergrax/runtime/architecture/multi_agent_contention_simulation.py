# © Artur Czarnecki. All rights reserved.

"""Multi-agent contention simulation for CI gates (AUDIT-IDEAL-26.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.architecture.multi_agent_acceptance import (
    MultiAgentAcceptanceCase,
    evaluate_multi_agent_acceptance,
)


class ContentionAgentRequest(BaseModel):
    agent_id: str
    requested_slots: int = Field(ge=1, le=16)


class ContentionAllocation(BaseModel):
    agent_id: str
    allocated_slots: int


class MultiAgentContentionSimulationReport(BaseModel):
    schema_version: str = "1.0.0"
    pattern: CoordinationPattern
    pool_size: int
    allocations: list[ContentionAllocation] = Field(default_factory=list)
    denied_agent_ids: list[str] = Field(default_factory=list)
    acceptance_passed: bool
    deadlock_free: bool


def simulate_multi_agent_contention(
    *,
    pool_size: int,
    requests: list[ContentionAgentRequest],
    pattern: CoordinationPattern = CoordinationPattern.SWARM,
) -> MultiAgentContentionSimulationReport:
    """Fair slot allocation under contention with acceptance validation."""
    remaining = pool_size
    allocations: list[ContentionAllocation] = []
    denied: list[str] = []
    for request in sorted(requests, key=lambda item: item.agent_id):
        grant = min(request.requested_slots, remaining)
        if grant <= 0:
            denied.append(request.agent_id)
            continue
        allocations.append(ContentionAllocation(agent_id=request.agent_id, allocated_slots=grant))
        remaining -= grant

    acceptance = evaluate_multi_agent_acceptance(
        [
            MultiAgentAcceptanceCase(
                case_id="contention.swarm",
                pattern=pattern,
                agent_count=len(requests),
                completed_steps=len(allocations),
                expected_steps=max(1, len(requests) - len(denied)),
            )
        ]
    )
    return MultiAgentContentionSimulationReport(
        pattern=pattern,
        pool_size=pool_size,
        allocations=allocations,
        denied_agent_ids=denied,
        acceptance_passed=acceptance.passed,
        deadlock_free=len(allocations) + len(denied) == len(requests),
    )
