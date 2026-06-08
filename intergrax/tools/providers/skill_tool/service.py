# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.providers.skill_tool.contracts import SkillResolveInput, SkillResolveOutput
from intergrax.tools.registry.runtime_bindings import SkillResolverBinding
from intergrax.tools.registry.wiring import ToolWiringContext

SKILL_RESOLVE_TOOL_ID = "skill.resolve"


def _require_skill_resolver(ctx: ToolWiringContext) -> SkillResolverBinding:
    resolver = ctx.skill_resolver or ctx.extras.get("skill_resolver")
    if resolver is None:
        raise RuntimeError("skill_resolver_not_configured")
    if not isinstance(resolver, SkillResolverBinding):
        raise RuntimeError("skill_resolver_invalid_type")
    return resolver


def skill_resolve(ctx: ToolWiringContext, params: SkillResolveInput) -> SkillResolveOutput:
    resolver = _require_skill_resolver(ctx)
    skill_ids = [item.strip() for item in params.skill_ids if item.strip()]
    pack = resolver.resolve(skill_ids)
    risk_tier = ""
    if hasattr(pack, "risk_tier"):
        risk = pack.risk_tier
        risk_tier = risk.value if hasattr(risk, "value") else str(risk)
    return SkillResolveOutput(
        skill_ids=list(pack.skill_ids) if hasattr(pack, "skill_ids") else skill_ids,
        tool_ids=sorted(pack.tool_ids) if hasattr(pack, "tool_ids") else [],
        prompt_instruction_ids=sorted(pack.prompt_instruction_ids) if hasattr(pack, "prompt_instruction_ids") else [],
        policy_fragment_ids=sorted(pack.policy_fragment_ids) if hasattr(pack, "policy_fragment_ids") else [],
        risk_tier=risk_tier,
    )
