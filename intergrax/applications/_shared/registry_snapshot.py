# © Artur Czarnecki. All rights reserved.

"""Materialized harness registry snapshot — compatibility re-export."""

from __future__ import annotations

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.registry.harness_snapshot import HarnessRegistrySnapshot


def resolve_registry_snapshot(
    ctx: ApplicationBuildContext,
    *,
    agent_registry: AgentRegistry | None = None,
    evaluation_registry: OnlineEvaluationRegistry | None = None,
) -> HarnessRegistrySnapshot:
    """Build typed registry snapshot from :class:`ApplicationBuildContext`."""
    return HarnessRegistrySnapshot(
        integration_profile=ctx.integration_profile,
        tool_registry=ctx.tool_registry,
        skill_registry=ctx.skill_registry,
        prompt_registry=ctx.prompt_registry,
        policy_bundle=ctx.policy_bundle,
        agent_registry=agent_registry,
        evaluation_registry=evaluation_registry,
    )


__all__ = ["HarnessRegistrySnapshot", "resolve_registry_snapshot"]
