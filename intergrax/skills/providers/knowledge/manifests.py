# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

KNOWLEDGE_OPENAI_STRICT = SkillManifest(
    skill_id="knowledge.openai_strict",
    version="1.0.0",
    description=(
        "Strict grounded Q&A via OpenAI managed vector store file_search "
        "(vendor-hosted retrieval, not harness rag.retrieve)."
    ),
    tool_ids=("openai.file_search.query",),
    prompt_instruction_ids=("knowledge.openai_strict.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("knowledge", "openai", "file_search", "strict_rag"),
)

KNOWLEDGE_WIKI_NAVIGATOR = SkillManifest(
    skill_id="knowledge.wiki_navigator",
    version="1.0.0",
    description="Internal wiki navigation: knowledge search, page fetch, and Confluence search.",
    tool_ids=("knowledge.search", "knowledge.get_page", "confluence.search"),
    prompt_instruction_ids=("knowledge.wiki_navigator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("knowledge", "wiki", "confluence"),
)
