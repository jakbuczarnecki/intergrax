# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

DEV_ISSUE_TRIAGE = SkillManifest(
    skill_id="dev.issue_triage",
    version="1.0.0",
    description="Provider-agnostic issue tracker search, read, comment, and notify.",
    tool_ids=("issues.search", "issues.get_issue", "issues.add_comment", "notify.send"),
    prompt_instruction_ids=("dev.issue_triage.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("dev", "issues", "triage"),
)

DEV_ISSUE_CREATOR = SkillManifest(
    skill_id="dev.issue_creator",
    version="1.0.0",
    description="Create new tracker issues from agent findings with search dedup and notify.",
    tool_ids=("issues.create_issue", "issues.search", "notify.send"),
    prompt_instruction_ids=("dev.issue_creator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("dev", "issues", "create"),
)
