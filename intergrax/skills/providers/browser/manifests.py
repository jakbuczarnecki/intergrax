# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

BROWSER_RESEARCH_FETCH = SkillManifest(
    skill_id="browser.research_fetch",
    version="1.0.0",
    description="Browser page fetch with URL read and document parse preview for research.",
    tool_ids=("browser.fetch_page", "websearch.read_url", "document.parse_preview"),
    prompt_instruction_ids=("browser.research_fetch.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("browser", "research", "fetch"),
)
