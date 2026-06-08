# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

GITLAB_ISSUE_CREATOR = SkillManifest(
    skill_id="gitlab.issue_creator",
    version="1.0.0",
    description="GitLab issue creation with dedup search and stakeholder notification.",
    tool_ids=("gitlab.create_issue", "issues.search", "notify.send"),
    prompt_instruction_ids=("gitlab.issue_creator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("gitlab", "issues", "create"),
)

