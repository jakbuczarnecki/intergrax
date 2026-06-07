# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.applications.contracts.manifest import AgentBinding
from dispute_analyst.dispute_analyst_agent import DisputeAnalystAgent
from dispute_intake.dispute_intake_agent import DisputeIntakeAgent
from dispute_scenario.dispute_scenario_agent import DisputeScenarioAgent
from dispute_strategist.dispute_strategist_agent import DisputeStrategistAgent
from dispute_sim_application.host.agent_builders import DISPUTE_SIM_AGENT_BUILDERS


def _build_from_context(
    agent_cls: type,
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> Agent:
    _ = ctx, binding
    factory = DISPUTE_SIM_AGENT_BUILDERS.get(agent_cls)
    if factory is None:
        raise ValueError(f"No builder registered for {binding.import_path!r}")
    return factory(ctx, binding)


def build_dispute_sim_dispute_intake_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> DisputeIntakeAgent:
    return _build_from_context(DisputeIntakeAgent, ctx, binding)


def build_dispute_sim_dispute_analyst_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> DisputeAnalystAgent:
    return _build_from_context(DisputeAnalystAgent, ctx, binding)


def build_dispute_sim_dispute_strategist_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> DisputeStrategistAgent:
    return _build_from_context(DisputeStrategistAgent, ctx, binding)


def build_dispute_sim_dispute_scenario_from_context(
    ctx: ApplicationBuildContext,
    binding: AgentBinding,
) -> DisputeScenarioAgent:
    return _build_from_context(DisputeScenarioAgent, ctx, binding)
