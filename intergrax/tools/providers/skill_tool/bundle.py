# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.providers.skill_tool.contracts import SkillResolveInput, SkillResolveOutput
from intergrax.tools.providers.skill_tool.handlers import SkillResolveHandler
from intergrax.tools.providers.skill_tool.service import SKILL_RESOLVE_TOOL_ID
from intergrax.tools.registry.runtime import ToolRegistry
from intergrax.tools.registry.wiring import ToolWiringContext

SKILL_BUNDLE_ID = "skill"
SKILL_TOOL_IDS: tuple[str, ...] = (SKILL_RESOLVE_TOOL_ID,)


def register_skill_tools(registry: ToolRegistry, ctx: ToolWiringContext) -> None:
    registry.register(
        ToolContract(
            tool_id=SKILL_RESOLVE_TOOL_ID,
            name=SKILL_RESOLVE_TOOL_ID,
            description="Resolve skill_ids into merged tool_ids, prompt refs, and policy fragments.",
            description_short="Resolve skills.",
            input_schema=SkillResolveInput,
            output_schema=SkillResolveOutput,
            error_mapping={},
            side_effects=False,
            category="skill",
            risk_level=ToolRiskLevel.LOW,
            tags=("skill", "introspection", "dx"),
        ),
        SkillResolveHandler(ctx),
    )
