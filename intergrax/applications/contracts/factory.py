# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent factory protocol for Tier-3 application wiring (Phase N.2.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Protocol

from intergrax.agents.agent_contract import Agent

if TYPE_CHECKING:
    from intergrax.applications.contracts.build_context import ApplicationBuildContext
    from intergrax.applications.contracts.manifest import AgentBinding


class CanonicalAgentFactory(Protocol):
    """Strict production factory contract: ``(ctx, binding) -> Agent``."""

    def __call__(
        self,
        ctx: ApplicationBuildContext,
        binding: AgentBinding,
    ) -> Agent: ...


# Broad alias retained for dev/lab builders and legacy compatibility surfaces.
AgentFactory = Callable[..., Agent]


class SupportsAgentFactory(Protocol):
    """Callable that builds a Tier-2 agent for a manifest binding."""

    def __call__(self, ctx: object, binding: object) -> Agent: ...
