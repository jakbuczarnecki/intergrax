# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Non-production Tier-2 agent registry bootstrap for lab, tests, and release tooling."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

from intergrax.runtime.architecture.capability_graph import CapabilityGraph, build_catalog_capability_graph
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.utils import attribute_access


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


def build_reference_baseline_agent_registries() -> tuple[AgentRegistry, ...]:
    """Union of harness reference agent families for baseline capability-graph tooling."""
    return (
        build_harness_registry(),
        build_research_registry(),
        build_legal_registry(),
        build_organization_worker_registry(),
    )


def build_reference_catalog_capability_graph() -> CapabilityGraph:
    """Baseline catalog graph with reference agent contracts (non-production adapter)."""
    return build_catalog_capability_graph(
        agent_registries=build_reference_baseline_agent_registries(),
    )


def merge_reference_agent_registries(
    registries: Sequence[AgentRegistry],
) -> AgentRegistry:
    """Merge multiple reference registries into one AgentRegistry."""
    merged = AgentRegistry()
    for registry in registries:
        for agent_id in registry.list_agent_ids():
            merged.register(registry.get(agent_id))
    return merged
