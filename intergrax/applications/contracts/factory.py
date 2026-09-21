# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Agent factory protocol for Tier-3 application wiring (Phase N.2.1)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

from intergrax.contracts.tier2_agent import Tier2Agent

if TYPE_CHECKING:
    from intergrax.applications.contracts.build_context import ApplicationBuildContext
    from intergrax.applications.contracts.manifest import AgentBinding

TSettings = TypeVar("TSettings")


class CanonicalAgentFactory(Protocol, Generic[TSettings]):
    """Strict production factory contract: ``(ctx, binding) -> Tier2Agent``."""

    def __call__(
        self,
        ctx: ApplicationBuildContext[TSettings],
        binding: AgentBinding,
    ) -> Tier2Agent: ...


# Public ABI name — structural :class:`CanonicalAgentFactory` (EBH-2D-C-R2).
AgentFactory = CanonicalAgentFactory

__all__ = ["AgentFactory", "CanonicalAgentFactory"]
