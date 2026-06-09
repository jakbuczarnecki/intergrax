# © Artur Czarnecki. All rights reserved.

"""Materialized harness registry snapshot from Tier-3 build context (Phase REG-1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.applications.contracts.build_context import ApplicationBuildContext
from intergrax.integrations.registry.profile import IntegrationProfile
from intergrax.prompts.registry.prompt_registry_protocol import PromptRegistryProtocol
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.policy.policy_bundle import RuntimePolicyBundle
from intergrax.runtime.registry.agent_registry import AgentRegistry
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
    agent_registry: AgentRegistry | None = None
    evaluation_registry: OnlineEvaluationRegistry | None = None
    explicit_prompt_bindings: dict[str, str] | None = None

    def tool_ids(self) -> tuple[str, ...]:
        if self.tool_registry is None:
            return ()
        return tuple(self.tool_registry.tool_ids())

    def skill_ids(self) -> tuple[str, ...]:
        if self.skill_registry is None:
            return ()
        return tuple(self.skill_registry.skill_ids())

    def prompt_ids(self) -> tuple[str, ...]:
        if self.prompt_registry is None:
            return ()
        return tuple(self.prompt_registry.list_prompt_ids())

    def agent_contract_ids(self) -> tuple[str, ...]:
        if self.agent_registry is None:
            return ()
        return tuple(sorted(contract.id for contract in self.agent_registry.list_contracts()))

    def evaluation_registry_ids(self) -> tuple[str, ...]:
        if self.evaluation_registry is None:
            return ()
        return ("evaluation:runtime_quality",)

    def resolved_prompt_bindings(self) -> dict[str, str]:
        if self.explicit_prompt_bindings is not None:
            return dict(self.explicit_prompt_bindings)
        if self.agent_registry is None:
            return {}
        bindings: dict[str, str] = {}
        for contract in self.agent_registry.list_contracts():
            if contract.prompt_binding_id:
                bindings[contract.id] = contract.prompt_binding_id
        return bindings


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
