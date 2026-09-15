# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Non-production Tier-2 agent registry bootstrap for lab, tests, and release tooling."""

from __future__ import annotations

import importlib
from collections.abc import Iterable, Mapping
from typing import Any

from intergrax.agents.agent_contract import Agent
from intergrax.runtime.events.event_bus import RuntimeEventBus
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.utils import attribute_access


class AgentRegistryBootstrapIdentityError(ValueError):
    """Dictionary key does not match the agent contract's canonical id."""


def bootstrap_agent_registry_from_agents(
    agents: Mapping[str, Agent] | Iterable[Agent],
    *,
    skill_registry: SkillRegistry | None = None,
    tool_registry: ToolRegistry | None = None,
    event_bus: RuntimeEventBus | None = None,
) -> AgentRegistry:
    """Explicit non-production registry construction for lab, fixtures, and unit tests."""
    registry = AgentRegistry()
    if isinstance(agents, Mapping):
        for agent_id, agent in agents.items():
            contract = agent.get_contract()
            if contract.id != agent_id:
                raise AgentRegistryBootstrapIdentityError(
                    "agent registry bootstrap identity mismatch: "
                    f"dict key {agent_id!r} != contract.id {contract.id!r}"
                )
            registry.register(
                agent,
                contract=contract,
                skill_registry=skill_registry,
                tool_registry=tool_registry,
                event_bus=event_bus,
            )
        return registry
    for agent in agents:
        registry.register(
            agent,
            skill_registry=skill_registry,
            tool_registry=tool_registry,
            event_bus=event_bus,
        )
    return registry


def _load_agent_class(module_name: str, class_name: str) -> Any:
    """Load a Tier-2 agent class for explicit lab/test bootstrap only."""
    module = importlib.import_module(module_name)
    return attribute_access.optional(module, class_name)


def build_harness_registry(*, include_echo: bool = True) -> AgentRegistry:
    """
    Build a minimal registry for experimentation (§41).

    Registers EchoAgent by default for harness smoke tests.
    Requires the echo agent package to be installed / on PYTHONPATH.
    """
    registry = AgentRegistry()
    if include_echo:
        EchoAgent = _load_agent_class("echo.echo_agent", "EchoAgent")
        registry.register(EchoAgent())
    return registry


def build_research_registry(*, include_echo: bool = False) -> AgentRegistry:
    """Registry with Research + Summary agents for multi-agent pipeline experiments."""
    ResearchAgent = _load_agent_class("research.research_agent", "ResearchAgent")
    SummaryAgent = _load_agent_class("research.summary_agent", "SummaryAgent")

    registry = AgentRegistry()
    registry.register(ResearchAgent())
    registry.register(SummaryAgent())
    if include_echo:
        EchoAgent = _load_agent_class("echo.echo_agent", "EchoAgent")
        registry.register(EchoAgent())
    return registry


def build_legal_registry() -> AgentRegistry:
    """Registry with Legal agent for capability-graph reference hosts."""
    LegalAgent = _load_agent_class("legal.legal_agent", "LegalAgent")

    registry = AgentRegistry()
    registry.register(LegalAgent())
    return registry


def build_organization_worker_registry(*, include_echo: bool = False) -> AgentRegistry:
    """Registry for §38 Organization Worker lab demos."""
    OrganizationWorkerAgent = _load_agent_class(
        "organization_worker.organization_worker_agent",
        "OrganizationWorkerAgent",
    )

    registry = AgentRegistry()
    registry.register(OrganizationWorkerAgent())
    if include_echo:
        EchoAgent = _load_agent_class("echo.echo_agent", "EchoAgent")
        registry.register(EchoAgent())
    return registry
