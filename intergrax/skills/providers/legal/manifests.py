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
