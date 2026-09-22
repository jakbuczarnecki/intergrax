# © Artur Czarnecki. All rights reserved.

"""Canonical agent factory for indirect_prompt_injection lab manifest wiring."""

from __future__ import annotations

from intergrax.agents.agent_contract import Agent
from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.factory import CanonicalAgentFactory
from intergrax.applications.contracts.manifest import AgentBinding
from intergrax.tools.registry import ToolRegistry

from platform_proofs.scenarios.indirect_prompt_injection.application.agent import (
    OrderAssistantAgent,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.runtime_composition import (
    ScenarioRuntimeComposition,
)
from platform_proofs.scenarios.indirect_prompt_injection.application.workflows import WorkflowKind


def build_order_assistant_agent_factory(
    *,
    registry: ToolRegistry,
    runtime_composition: ScenarioRuntimeComposition,
    workflow: WorkflowKind,
    order_id: str = "48291",
    user_message: str = "",
) -> CanonicalAgentFactory:
    """Return a factory that materializes OrderAssistantAgent with runtime dependencies."""

    def _factory(
        ctx: ApplicationBuildContext[object],
        binding: AgentBinding,
    ) -> Agent:
        del ctx, binding
        return OrderAssistantAgent(
            registry=registry,
            runtime_composition=runtime_composition,
            workflow=workflow,
            order_id=order_id,
            user_message=user_message,
        )

    return _factory


def bind_order_assistant_agent(agent: OrderAssistantAgent) -> CanonicalAgentFactory:
    """Bind an already materialized lab agent for manifest factory conformance."""

    def _factory(
        ctx: ApplicationBuildContext[object],
        binding: AgentBinding,
    ) -> Agent:
        del ctx, binding
        return agent

    return _factory


__all__ = [
    "bind_order_assistant_agent",
    "build_order_assistant_agent_factory",
]
