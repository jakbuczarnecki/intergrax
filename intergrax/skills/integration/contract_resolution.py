# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolutionError, SkillResolver
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.runtime import ToolRegistry


def _extra_tool_ids(extra_tools: list[ToolContract]) -> list[str]:
    return [tool.tool_id for tool in extra_tools]


def resolve_contract_tools(
    contract: AgentContract,
    *,
    skill_resolver: SkillResolver,
    tool_registry: ToolRegistry | None = None,
) -> tuple[AgentContract, ResolvedSkillPack]:
    """
    Merge ``contract.skills`` and ``contract.extra_tools`` into ``allowed_tools``.

    Validates manifests and optional tool references when ``tool_registry`` is provided.
    """
    resolver = (
        SkillResolver(skill_resolver.skill_registry, tool_registry)
        if tool_registry is not None
        else skill_resolver
    )
    if contract.skills:
        resolver.validate_skills(contract.skills)
    pack = resolver.resolve_skills(contract.skills)
    extra_ids = _extra_tool_ids(contract.extra_tools)
    if tool_registry is not None and extra_ids:
        resolver.validate_tool_contracts(contract.extra_tools)
    merged_tools = list(pack.merged_allowed_tools(extra_ids))
    updated = contract.model_copy(update={"allowed_tools": merged_tools})
    return updated, pack
