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

CACHE_KEY_ADMIN = SkillManifest(
    skill_id="cache.key_admin",
    version="1.0.0",
    description="Cache key administration: list, get, and delete session cache keys.",
    tool_ids=("cache.list_keys", "cache.get", "cache.delete"),
    prompt_instruction_ids=("cache.key_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("cache", "admin", "keys"),
)

CACHE_WARM_PREFETCH = SkillManifest(
    skill_id="cache.warm_prefetch",
    version="1.0.0",
    description="Warm session cache from retrieval results.",
    tool_ids=("cache.set", "cache.get", "rag.retrieve"),
    prompt_instruction_ids=("cache.warm_prefetch.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("cache", "warm", "prefetch"),
)

