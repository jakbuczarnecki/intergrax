# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CONTEXT_TOKEN_PLANNER = SkillManifest(
    skill_id="context.token_planner",
    version="1.0.0",
    description="Context budget planning: estimate tokens, summarize overflow, and read session memory.",
    tool_ids=("context.estimate_tokens", "context.summarize", "memory.read"),
    prompt_instruction_ids=("context.token_planner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("context", "tokens", "budget"),
)
