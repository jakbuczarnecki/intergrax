# © Artur Czarnecki. All rights reserved.

"""Shared NPSC-5D governance test adapters."""

from __future__ import annotations

from intergrax.runtime.governance.multi_agent_coordination_governance import (
    AllowingMultiAgentCoordinationGovernance,
    DenyingMultiAgentCoordinationGovernance,
    RequireHumanMultiAgentCoordinationGovernance,
    UnavailableMultiAgentCoordinationGovernance,
)


def allowing_coordination_governance() -> AllowingMultiAgentCoordinationGovernance:
    return AllowingMultiAgentCoordinationGovernance()


def denying_coordination_governance() -> DenyingMultiAgentCoordinationGovernance:
    return DenyingMultiAgentCoordinationGovernance()


def require_human_coordination_governance() -> RequireHumanMultiAgentCoordinationGovernance:
    return RequireHumanMultiAgentCoordinationGovernance()


def unavailable_coordination_governance() -> UnavailableMultiAgentCoordinationGovernance:
    return UnavailableMultiAgentCoordinationGovernance()
