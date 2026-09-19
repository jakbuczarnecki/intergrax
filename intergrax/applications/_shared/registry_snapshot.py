# © Artur Czarnecki. All rights reserved.

"""Materialized harness registry snapshot — compatibility re-export."""

from __future__ import annotations

from intergrax.applications._shared.application_composition_context import (
    ApplicationCompositionContext,
)
from intergrax.runtime.registry.agent_registry import AgentRegistry
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.registry.harness_snapshot import HarnessRegistrySnapshot


def resolve_registry_snapshot(
    composition: ApplicationCompositionContext,
    *,
    agent_registry: AgentRegistry | None = None,
    evaluation_registry: OnlineEvaluationRegistry | None = None,
) -> HarnessRegistrySnapshot:
    """Build typed registry snapshot from private composition context."""
    return HarnessRegistrySnapshot(
        integration_profile=composition.integration_profile,
        tool_registry=composition.tool_registry,
        skill_registry=composition.skill_registry,
        prompt_registry=composition.prompt_registry,
        policy_bundle=composition.policy_bundle,
        agent_registry=agent_registry,
        evaluation_registry=evaluation_registry,
    )


__all__ = ["HarnessRegistrySnapshot", "resolve_registry_snapshot"]
