# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.policy.tool_policy_resolution import resolve_allowed_tools
from intergrax.runtime.tools.scope_policy import StaticToolScopePolicy
from intergrax.skills.core.contracts import SkillManifest
from intergrax.skills.registry.profile import SkillProfile, is_skill_enabled
from intergrax.skills.registry.tool_requirements import available_tool_ids_for_profile
from intergrax.skills.resolver import ResolvedSkillPack, SkillResolutionError, SkillResolverProtocol
from intergrax.tools.core.contracts import ToolContract
from intergrax.tools.registry.profile import ToolProfile
from intergrax.tools.registry.runtime import ToolRegistry


def _extra_tool_ids(extra_tools: list[ToolContract]) -> list[str]:
    return [tool.tool_id for tool in extra_tools]


def eligible_agent_skill_manifests(
    skills: Sequence[SkillManifest],
    skill_profile: SkillProfile | None,
) -> tuple[SkillManifest, ...]:
    """
    Agent-declared skills gated by host ``SkillProfile`` enablement.

    When ``skill_profile`` is absent (explicit dev/test registry), manifests pass through.
    """
    if not skills:
        return ()
    if skill_profile is None:
        return tuple(skills)
    eligible: list[SkillManifest] = []
    for manifest in skills:
        skill_id = manifest.skill_id.strip()
        if not skill_id:
            raise SkillResolutionError("SkillManifest.skill_id must be non-empty")
        if not is_skill_enabled(skill_profile, skill_id):
            raise SkillResolutionError(
                f"skill not enabled for host SkillProfile: {skill_id}",
            )
        eligible.append(manifest)
    return tuple(eligible)


def clamp_tool_candidates_to_profile(
    tool_ids: Sequence[str],
    tool_profile: ToolProfile | None,
) -> list[str]:
    """Fail closed when any candidate exceeds host ``ToolProfile`` availability."""
    normalized = [item.strip() for item in tool_ids if item.strip()]
    if tool_profile is None or not normalized:
        return sorted(normalized)
    available = frozenset(available_tool_ids_for_profile(tool_profile))
    violations = sorted({tool_id for tool_id in normalized if tool_id not in available})
    if violations:
        raise SkillResolutionError(
            "tool candidates exceed host ToolProfile availability: "
            + ", ".join(violations),
        )
    return sorted(normalized)


def resolve_gateway_allowed_tools(
    contract_allowed_tools: Sequence[str],
    metadata_allowed_tools: object,
) -> list[str]:
    """
    Narrow-only merge for ACP / step metadata ``allowed_tools``.

    Untrusted metadata cannot widen the contract-derived ceiling. An empty contract
    ceiling stays empty — metadata cannot restore tools.
    """
    ceiling = tuple(
        tool_id.strip() for tool_id in contract_allowed_tools if tool_id.strip()
    )
    if metadata_allowed_tools is None:
        return list(ceiling)
    if not isinstance(metadata_allowed_tools, list):
        raise SkillResolutionError(
            "allowed_tools metadata must be a list when present",
        )
    metadata_ids = [
        str(tool_id).strip()
        for tool_id in metadata_allowed_tools
        if str(tool_id).strip()
    ]
    if not ceiling:
        return []
    if not metadata_ids:
        return list(ceiling)
    upstream = StaticToolScopePolicy(allowed_tools=set(ceiling))
    resolved = resolve_allowed_tools(upstream_policy=upstream, explicit=metadata_ids)
    if resolved is None:
        return list(ceiling)
    return list(resolved)


def resolve_contract_tools(
    contract: AgentContract,
    *,
    skill_resolver: SkillResolverProtocol,
    tool_registry: ToolRegistry | None = None,
    skill_profile: SkillProfile | None = None,
    tool_profile: ToolProfile | None = None,
) -> tuple[AgentContract, ResolvedSkillPack]:
    """
    Merge ``contract.skills`` and ``contract.extra_tools`` into ``allowed_tools``.

    Skills are gated by host ``SkillProfile`` when provided. Tool candidates are
    clamped to host ``ToolProfile`` when provided. Validates manifests and optional
    tool references when ``tool_registry`` is provided.
    """
    from intergrax.skills.resolver import SkillResolver

    resolver: SkillResolverProtocol = (
        SkillResolver(skill_resolver.skill_registry, tool_registry)
        if tool_registry is not None
        else skill_resolver
    )
    eligible_skills = eligible_agent_skill_manifests(contract.skills, skill_profile)
    if eligible_skills:
        resolver.validate_skills(eligible_skills)
    pack = resolver.resolve_skills(eligible_skills)
    extra_ids = _extra_tool_ids(contract.extra_tools)
    if tool_registry is not None and extra_ids:
        resolver.validate_tool_contracts(contract.extra_tools)
    merged_tools = clamp_tool_candidates_to_profile(
        pack.merged_allowed_tools(extra_ids),
        tool_profile,
    )
    updated = contract.model_copy(update={"allowed_tools": merged_tools})
    return updated, pack
