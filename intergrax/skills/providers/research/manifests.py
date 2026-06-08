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

RESEARCH_WEB_CACHE_ADMIN = SkillManifest(
    skill_id="research.web_cache_admin",
    version="1.0.0",
    description="Web search cache admin: invalidate cache, query, and batch fetch.",
    tool_ids=("websearch.invalidate_cache", "websearch.query", "websearch.fetch_batch"),
    prompt_instruction_ids=("research.web_cache_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("research", "web", "cache"),
)

RESEARCH_DEEP_DIVE = SkillManifest(
    skill_id="research.deep_dive",
    version="1.0.0",
    description="Deep web research with batch fetch and report workspace export.",
    tool_ids=("websearch.fetch_batch", "websearch.read_url", "workspace.write_file"),
    prompt_instruction_ids=("research.deep_dive.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "deep_dive", "web"),
)


RESEARCH_SOURCE_VALIDATOR = SkillManifest(
    skill_id="research.source_validator",
    version="1.0.0",
    description="Validate sources against index and parse previews.",
    tool_ids=("websearch.query", "rag.retrieve", "document.parse_preview"),
    prompt_instruction_ids=("research.source_validator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "source", "validator"),
)


RESEARCH_REPORT_COMPILER = SkillManifest(
    skill_id="research.report_compiler",
    version="1.0.0",
    description="Compile citation-backed reports from retrieval and web evidence.",
    tool_ids=("rag.retrieve", "websearch.query", "workspace.write_file"),
    prompt_instruction_ids=("research.report_compiler.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("research", "report", "compiler"),
)

