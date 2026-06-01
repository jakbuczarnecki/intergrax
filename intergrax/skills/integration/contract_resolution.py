# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolver
from intergrax.tools.registry.runtime import ToolRegistry


def resolve_contract_tools(
    contract: AgentContract,
    *,
    skill_resolver: SkillResolver,
    tool_registry: ToolRegistry | None = None,
) -> tuple[AgentContract, ResolvedSkillPack]:
    """
    Merge ``contract.skill_ids`` into ``allowed_tools`` and return updated contract copy.

    Validates skill_ids and optional tool references when ``tool_registry`` is provided.
    """
    resolver = (
        SkillResolver(skill_resolver.skill_registry, tool_registry)
        if tool_registry is not None
        else skill_resolver
    )
    if contract.skill_ids:
        resolver.validate_skill_ids(contract.skill_ids)
    pack = resolver.resolve(contract.skill_ids)
    merged_tools = list(pack.merged_allowed_tools(contract.allowed_tools))
    updated = contract.model_copy(update={"allowed_tools": merged_tools})
    return updated, pack
