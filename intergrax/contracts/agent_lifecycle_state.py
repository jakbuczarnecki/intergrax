# © Artur Czarnecki. All rights reserved.

"""Agent lifecycle state enum shared by contracts and governance evaluators."""

from __future__ import annotations

from enum import Enum


class AgentLifecycleState(str, Enum):  # noqa: UP042  (str-mixin kept: StrEnum would change str()/format() of a shared contract)
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


ROUTING_INELIGIBLE_LIFECYCLE_STATES: frozenset[AgentLifecycleState] = frozenset(
    {AgentLifecycleState.DEPRECATED, AgentLifecycleState.RETIRED}
)


def agent_lifecycle_state_is_routable(state: AgentLifecycleState) -> bool:
    """False for deprecated and retired agents: they never take part in Nexus selection."""
    return state not in ROUTING_INELIGIBLE_LIFECYCLE_STATES


def audit_map_lifecycle_label(state: AgentLifecycleState) -> str:
    """Map harness lifecycle states to IDEAL audit-map vocabulary."""
    mapping = {
        AgentLifecycleState.EXPERIMENTAL: "draft",
        AgentLifecycleState.DEVELOPMENT: "experimental",
        AgentLifecycleState.CANDIDATE: "candidate",
        AgentLifecycleState.STAGING: "candidate",
        AgentLifecycleState.PRODUCTION: "certified",
        AgentLifecycleState.DEPRECATED: "deprecated",
        AgentLifecycleState.RETIRED: "retired",
    }
    return mapping[state]
