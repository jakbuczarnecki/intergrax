# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

LEGAL_CONTRACT_REVIEW = SkillManifest(
    skill_id="legal.contract_review",
    version="1.0.0",
    description="Contract review capability: retrieval, web evidence, and legal tool planning.",
    tool_ids=("rag.retrieve", "websearch.query"),
    prompt_instruction_ids=("legal.contract_review.system",),
    policy_fragment_id="legal.contract_review.policy",
    risk_tier=SkillRiskTier.HIGH,
    tags=("legal", "contract", "review"),
)

LEGAL_CLAUSE_COMPARE = SkillManifest(
    skill_id="legal.clause_compare",
    version="1.0.0",
    description="Compare contract clauses with retrieval, workspace drafts, and web evidence.",
    tool_ids=("rag.retrieve", "workspace.write_file", "websearch.query"),
    prompt_instruction_ids=("legal.clause_compare.system",),
    policy_fragment_id="legal.contract_review.policy",
    risk_tier=SkillRiskTier.HIGH,
    requires_skills=("legal.contract_review",),
    tags=("legal", "clause", "compare"),
)

LEGAL_CASE_RESEARCH = SkillManifest(
    skill_id="legal.case_research",
    version="1.0.0",
    description="Case and regulatory research: index retrieval, wiki search, and web evidence.",
    tool_ids=("rag.retrieve", "knowledge.search", "websearch.query"),
    prompt_instruction_ids=("legal.case_research.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("legal", "case", "research"),
)
