# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

OPENAI_VECTOR_ADMIN = SkillManifest(
    skill_id="openai.vector_admin",
    version="1.0.0",
    description="OpenAI vector store admin: upload, clear, and file_search query.",
    tool_ids=("openai.vector_store.upload", "openai.vector_store.clear", "openai.file_search.query"),
    prompt_instruction_ids=("openai.vector_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("openai", "vector_store", "admin"),
)

