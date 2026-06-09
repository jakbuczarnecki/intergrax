# © Artur Czarnecki. All rights reserved.

"""Harness artifact lifecycle states (IDEAL-19.1)."""

from __future__ import annotations

from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState

# Shared lifecycle vocabulary for agents, tools, skills, prompts, and integrations.
ArtifactLifecycleState = AgentLifecycleState

BLOCKED_RESOLUTION_STATES: frozenset[ArtifactLifecycleState] = frozenset(
    {
        ArtifactLifecycleState.RETIRED,
    }
)


def is_resolution_allowed(state: ArtifactLifecycleState) -> bool:
    """Return False when an artifact must not be resolved for new runs."""
    return state not in BLOCKED_RESOLUTION_STATES
