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

RESEARCH_WEB_EVIDENCE = SkillManifest(
    skill_id="research.web_evidence",
    version="1.0.0",
    description="Web-grounded evidence: search, single URL read, and batch fetch.",
    tool_ids=("websearch.query", "websearch.read_url", "websearch.fetch_batch"),
    prompt_instruction_ids=("research.web_evidence.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "web", "evidence"),
)

RESEARCH_CITATION_SYNTHESIS = SkillManifest(
    skill_id="research.citation_synthesis",
    version="1.0.0",
    description="Citation-backed synthesis: retrieval, web search, and workspace report export.",
    tool_ids=("rag.retrieve", "websearch.query", "workspace.write_file"),
    prompt_instruction_ids=("research.citation_synthesis.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "citation", "synthesis"),
)
