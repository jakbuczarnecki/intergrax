# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CACHE_SESSION_CACHE = SkillManifest(
    skill_id="cache.session_cache",
    version="1.0.0",
    description="Key-value cache layer with task memory read for session-scoped acceleration.",
    tool_ids=("cache.get", "cache.set", "memory.read"),
    prompt_instruction_ids=("cache.session_cache.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("cache", "session", "kv"),
)
