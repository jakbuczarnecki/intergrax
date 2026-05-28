# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.runtime.registry.agent_registry import AgentRegistry


def build_harness_registry(*, include_echo: bool = True) -> AgentRegistry:
    """
    Build a minimal registry for experimentation (§41).

    Registers EchoAgent by default for harness smoke tests.
    """
    registry = AgentRegistry()
    if include_echo:
        from echo.echo_agent import EchoAgent

        registry.register(EchoAgent())
    return registry


def build_research_registry(*, include_echo: bool = False) -> AgentRegistry:
    """Registry with Research + Summary agents for multi-agent pipeline experiments."""
    from research.research_agent import ResearchAgent
    from research.summary_agent import SummaryAgent

    registry = AgentRegistry()
    registry.register(ResearchAgent())
    registry.register(SummaryAgent())
    if include_echo:
        from echo.echo_agent import EchoAgent

        registry.register(EchoAgent())
    return registry


def build_organization_worker_registry(*, include_echo: bool = False) -> AgentRegistry:
    """Registry for §38 Organization Worker lab demos."""
    from organization_worker.organization_worker_agent import OrganizationWorkerAgent

    registry = AgentRegistry()
    registry.register(OrganizationWorkerAgent())
    if include_echo:
        from echo.echo_agent import EchoAgent

        registry.register(EchoAgent())
    return registry
