# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

LOCAL_WORKSPACE_INDEX = SkillManifest(
    skill_id="local.workspace.index",
    version="1.0.0",
    description="Index user-local source paths into tenant-scoped RAG vector store.",
    tool_ids=("rag.ingest_document",),
    prompt_instruction_ids=("local.workspace.index.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("local", "workspace", "index", "rag"),
)

LOCAL_WORKSPACE_SEARCH = SkillManifest(
    skill_id="local.workspace.search",
    version="1.0.0",
    description="Semantic search and tenant-scoped evidence retrieval over locally indexed documents.",
    tool_ids=("rag.retrieve",),
    prompt_instruction_ids=("local.workspace.search.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("local", "workspace", "search", "rag"),
)

LOCAL_WORKSPACE_SYNTHESIZE = SkillManifest(
    skill_id="local.workspace.synthesize",
    version="1.0.0",
    description="Synthesize reports and drafts from evidence; write only to shadow workspace.",
    tool_ids=("workspace.write_file",),
    prompt_instruction_ids=("local.workspace.synthesize.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("local", "workspace", "synthesize", "shadow"),
)
