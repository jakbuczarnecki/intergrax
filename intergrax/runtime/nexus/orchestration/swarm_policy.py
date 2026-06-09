# © Artur Czarnecki. All rights reserved.

"""Swarm coordination policy (ORCH-5.1 / CFG-17)."""

from __future__ import annotations

from intergrax.runtime.architecture.multi_agent_coordination import CoordinationPattern
from intergrax.runtime.nexus.planning.task_planner import NexusPlan


class SwarmCoordinationError(ValueError):
    """Raised when a swarm-labelled plan violates platform constraints."""


def annotate_plan_coordination_pattern(
    plan: NexusPlan,
    *,
    coordination_pattern: str | None,
) -> NexusPlan:
    if not coordination_pattern:
        return plan
    metadata = dict(plan.plan_metadata)
    metadata["coordination_pattern"] = coordination_pattern
    annotated = plan.model_copy(update={"plan_metadata": metadata})
    if coordination_pattern == CoordinationPattern.SWARM.value:
        validate_swarm_plan(annotated)
    return annotated


def validate_swarm_plan(plan: NexusPlan, *, min_parallel_roots: int = 3) -> None:
    """
    Swarm (D7) requires at least ``min_parallel_roots`` root steps (no ``depends_on``).

    Parallel batches are derived from the execution graph; roots approximate swarm width.
    """
    roots = [step for step in plan.steps if not step.depends_on]
    if len(roots) < min_parallel_roots:
        raise SwarmCoordinationError(
            f"swarm pattern requires at least {min_parallel_roots} parallel root steps, "
            f"got {len(roots)}"
        )
