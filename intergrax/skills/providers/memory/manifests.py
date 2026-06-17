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

MEMORY_LTM_CURATOR = SkillManifest(
    skill_id="memory.ltm_curator",
    version="1.0.0",
    description="Long-term memory curation: write durable facts, search LTM, and read session context.",
    tool_ids=("ltm.write_fact", "ltm.search", "memory.read"),
    prompt_instruction_ids=("memory.ltm_curator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("memory", "ltm", "facts"),
)

MEMORY_SEMANTIC_SEARCH = SkillManifest(
    skill_id="memory.semantic_search",
    version="1.0.0",
    description="Semantic memory search across session episodic index and LTM vector index.",
    tool_ids=("memory.semantic_search", "ltm.search"),
    prompt_instruction_ids=("memory.semantic_search.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("memory", "semantic", "search"),
)

MEMORY_CROSS_TURN_NOTES = SkillManifest(
    skill_id="memory.cross_turn_notes",
    version="1.0.0",
    description="Cross-turn note taking with list/read/write task memory.",
    tool_ids=("memory.write", "memory.list_keys", "memory.read"),
    prompt_instruction_ids=("memory.cross_turn_notes.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("memory", "notes", "cross_turn"),
)


MEMORY_FACT_EXTRACTOR = SkillManifest(
    skill_id="memory.fact_extractor",
    version="1.0.0",
    description="Extract durable facts into LTM with context summarization.",
    tool_ids=("ltm.write_fact", "memory.read", "context.summarize"),
    prompt_instruction_ids=("memory.fact_extractor.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("memory", "fact", "extractor"),
)

