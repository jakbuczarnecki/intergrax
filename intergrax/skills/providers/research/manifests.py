# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

RESEARCH_LITERATURE_SCAN = SkillManifest(
    skill_id="research.literature_scan",
    version="1.0.0",
    description="Literature scan: hybrid retrieval and web evidence for research pipelines.",
    tool_ids=("rag.retrieve", "websearch.query"),
    prompt_instruction_ids=("research.literature_scan.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "literature", "retrieval"),
)
