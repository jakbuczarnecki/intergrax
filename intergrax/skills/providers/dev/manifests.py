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

DEV_ISSUE_UPDATER = SkillManifest(
    skill_id="dev.issue_updater",
    version="1.0.0",
    description="Update existing tracker issues: fetch, comment, and transition state.",
    tool_ids=("issues.update_issue", "issues.add_comment", "issues.get_issue"),
    prompt_instruction_ids=("dev.issue_updater.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("dev", "issues", "update"),
)

DEV_PR_REVIEWER = SkillManifest(
    skill_id="dev.pr_reviewer",
    version="1.0.0",
    description="PR/issue review with search, fetch, and mail notification.",
    tool_ids=("issues.search", "issues.get_issue", "collaboration.send_mail"),
    prompt_instruction_ids=("dev.pr_reviewer.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("dev", "pr", "review"),
)


DEV_RELEASE_NOTES = SkillManifest(
    skill_id="dev.release_notes",
    version="1.0.0",
    description="Release notes from issue search and workspace export.",
    tool_ids=("issues.search", "workspace.write_file", "notify.send"),
    prompt_instruction_ids=("dev.release_notes.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("dev", "release", "notes"),
)


DEV_SPRINT_PLANNER = SkillManifest(
    skill_id="dev.sprint_planner",
    version="1.0.0",
    description="Sprint planning with issues, calendar, and scratchpad memory.",
    tool_ids=("issues.search", "collaboration.list_calendar", "memory.write"),
    prompt_instruction_ids=("dev.sprint_planner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("dev", "sprint", "planner"),
)

