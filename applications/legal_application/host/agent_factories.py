# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed Tier-3 factories for the legal application."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from legal.legal_agent import LegalAgent
from legal_application.host.settings import LegalBackendSettings


def build_legal_agent_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> LegalAgent:
    """Canonical ``AgentFactory`` for :class:`~legal.legal_agent.LegalAgent`."""
    _ = binding
    settings = ctx.settings
    if not isinstance(settings, LegalBackendSettings):
        raise TypeError(
            "Legal agent factory requires LegalBackendSettings on ApplicationBuildContext"
        )
    from legal_application.host.wiring import build_legal_agent

    return build_legal_agent(settings)
