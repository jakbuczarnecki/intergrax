# © Artur Czarnecki. All rights reserved.

"""Nexus plan structural validation (COG-1.3 / ORCH-CONFIG.10)."""

from __future__ import annotations

from intergrax.runtime.nexus.planning.task_planner import NexusPlan
from intergrax.runtime.registry.agent_registry import AgentRegistry


def validate_nexus_plan(plan: NexusPlan, registry: AgentRegistry) -> list[str]:
    """
    Return human-readable validation errors for ``plan``.

    Checks unknown ``depends_on`` references and unregistered ``agent_id`` values.
    """
    if not plan.steps:
        return []

    errors: list[str] = []
    step_ids = {step.step_id for step in plan.steps}
    known_agents = set(registry.list_agent_ids())

    for step in plan.steps:
        for dep in step.depends_on:
            if dep not in step_ids:
                errors.append(
                    f"step {step.step_id!r}: unknown depends_on {dep!r}"
                )
        if step.agent_id and step.agent_id not in known_agents:
            errors.append(
                f"step {step.step_id!r}: unknown agent_id {step.agent_id!r}"
            )

    return errors
