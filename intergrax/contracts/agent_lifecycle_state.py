# © Artur Czarnecki. All rights reserved.

"""Agent lifecycle state enum shared by contracts and governance evaluators."""

from __future__ import annotations

from enum import Enum


class AgentLifecycleState(str, Enum):
    EXPERIMENTAL = "experimental"
    DEVELOPMENT = "development"
    CANDIDATE = "candidate"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    RETIRED = "retired"


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
