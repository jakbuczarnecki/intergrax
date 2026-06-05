# © Artur Czarnecki. All rights reserved.

"""Materialized harness registry snapshot from Tier-3 build context (Phase REG-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.skills.registry.runtime import SkillRegistry
from intergrax.tools.registry.runtime import ToolRegistry


@dataclass(frozen=True, slots=True)
class HarnessRegistrySnapshot:
    """Resolved registry handles available during Tier-3 host assembly."""

    integration_profile: IntegrationProfile | None
    tool_registry: ToolRegistry | None
    skill_registry: SkillRegistry | None
    prompt_registry: PromptRegistryProtocol | None
    policy_bundle: RuntimePolicyBundle | None

    def tool_ids(self) -> tuple[str, ...]:
        if self.tool_registry is None:
            return ()
        return tuple(self.tool_registry.tool_ids())

    def skill_ids(self) -> tuple[str, ...]:
        if self.skill_registry is None:
            return ()
        return tuple(self.skill_registry.skill_ids())


def resolve_registry_snapshot(ctx: ApplicationBuildContext) -> HarnessRegistrySnapshot:
    """Build typed registry snapshot from :class:`ApplicationBuildContext`."""
    return HarnessRegistrySnapshot(
        integration_profile=ctx.integration_profile,
        tool_registry=ctx.tool_registry,
        skill_registry=ctx.skill_registry,
        prompt_registry=ctx.prompt_registry,
        policy_bundle=ctx.policy_bundle,
    )
