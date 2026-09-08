# © Artur Czarnecki. All rights reserved.

"""In-memory capability grant resolver (DI-friendly, no global state)."""

from __future__ import annotations

from intergrax.contracts.agent_runtime_governance import (
    AgentIdentity,
    CapabilityGrant,
    CapabilityGrantResolverPort,
)


class InMemoryCapabilityGrantResolver:
    """Resolves capability grants from an explicit injected grant table."""

    def __init__(self, grants: tuple[CapabilityGrant, ...] = ()) -> None:
        self._grants = tuple(grants)

    def resolve_grant(self, agent: AgentIdentity) -> CapabilityGrant | None:
        for grant in self._grants:
            if grant.agent_id == agent.agent_id and grant.tenant_id == agent.tenant_id:
                return grant
        return None


def require_capability_granted(
    *,
    grant: CapabilityGrant | None,
    capability: str,
    run_id: str,
    agent_id: str,
    tool_id: str,
) -> None:
    """Fail closed when capability is not explicitly granted."""
    from intergrax.runtime.agent_governance.errors import CapabilityNotGrantedError

    if grant is None or not grant.is_capability_allowed(capability):
        raise CapabilityNotGrantedError(
            run_id=run_id,
            agent_id=agent_id,
            capability=capability,
            tool_id=tool_id,
        )
