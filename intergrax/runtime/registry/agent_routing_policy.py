# © Artur Czarnecki. All rights reserved.

"""Runtime agent routing policy (Phase V-REM-ALG.1, V-REM-ALG.2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_lifecycle_state import AgentLifecycleState


@dataclass(frozen=True, slots=True)
class AgentRoutingDecision:
    routable: bool
    reason: str | None = None


def evaluate_agent_routing(
    contract: AgentContract,
    *,
    production_mode: bool,
) -> AgentRoutingDecision:
    """Return whether an agent may participate in Nexus selection."""
    if contract.lifecycle_state in {
        AgentLifecycleState.DEPRECATED,
        AgentLifecycleState.RETIRED,
    }:
        return AgentRoutingDecision(
            routable=False,
            reason=f"agent lifecycle state is {contract.lifecycle_state.value}",
        )

    if not production_mode:
        return AgentRoutingDecision(routable=True)

    if contract.lifecycle_state not in {
        AgentLifecycleState.PRODUCTION,
        AgentLifecycleState.STAGING,
    }:
        return AgentRoutingDecision(
            routable=False,
            reason=(
                "production_mode requires lifecycle state production or staging; "
                f"got {contract.lifecycle_state.value}"
            ),
        )

    if not contract.production_eligible:
        return AgentRoutingDecision(routable=True)

    reasons: list[str] = []
    if not contract.owner_team or not contract.owner_contact:
        reasons.append("Missing owner metadata for production-eligible agent")
    if not contract.runbook_ref:
        reasons.append("Missing runbook reference for production-eligible agent")
    if reasons:
        return AgentRoutingDecision(routable=False, reason="; ".join(reasons))

    return AgentRoutingDecision(routable=True)
