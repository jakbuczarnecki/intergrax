# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

MEMORY_TASK_SCRATCHPAD = SkillManifest(
    skill_id="memory.task_scratchpad",
    version="1.0.0",
    description="Task-scoped key-value scratchpad for multi-step agent continuity.",
    tool_ids=("memory.read", "memory.write", "memory.list_keys"),
    prompt_instruction_ids=("memory.task_scratchpad.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("memory", "task", "scratchpad"),
)

MEMORY_SESSION_CLEANUP = SkillManifest(
    skill_id="memory.session_cleanup",
    version="1.0.0",
    description="Session memory hygiene: list keys, delete stale records, read before purge.",
    tool_ids=("memory.list_keys", "memory.delete_key", "memory.read"),
    prompt_instruction_ids=("memory.session_cleanup.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("memory", "cleanup", "session"),
)
