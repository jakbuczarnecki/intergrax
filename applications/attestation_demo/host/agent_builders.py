# © Artur Czarnecki. All rights reserved.

"""Type-keyed Tier-3 agent factories for attestation_demo."""

from __future__ import annotations

from boundary_demo.boundary_demo_agent import BoundaryDemoAgent
from intergrax.agents.agent_contract import Agent
from intergrax.applications._shared.application_composition_context import (
    optional_factory_composition,
)
from intergrax.applications._shared.lab_harness_context import lab_harness_context_from_build_context
from intergrax.applications.contracts.factory import AgentFactory
from intergrax.runtime.attestation.settings import resolve_execution_boundary_export_runtime


def _build_boundary_demo_agent(ctx, _binding) -> Agent:
    harness = lab_harness_context_from_build_context(ctx)
    export_settings = None
    if ctx.environment is not None:
        export_settings = resolve_execution_boundary_export_runtime(
            ctx.environment.execution_boundary_export_profile,
        )
    composition = optional_factory_composition()
    environment = ctx.environment
    tool_profile = (
        composition.tool_profile
        if composition is not None and composition.tool_profile is not None
        else (environment.tool_profile if environment is not None else None)
    )
    tool_wiring_context = (
        composition.tool_wiring_context if composition is not None else None
    )
    boundary_event_buffer = (
        composition.boundary_event_buffer if composition is not None else None
    )
    return BoundaryDemoAgent(
        harness,
        tool_profile=tool_profile,
        tool_wiring_context=tool_wiring_context,
        execution_boundary_export=export_settings,
        boundary_event_buffer=boundary_event_buffer,
    )


ATTESTATION_DEMO_AGENT_BUILDERS: dict[type[Agent], AgentFactory] = {
    BoundaryDemoAgent: _build_boundary_demo_agent,
}
