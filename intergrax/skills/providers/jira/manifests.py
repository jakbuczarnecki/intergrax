# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

JIRA_TASK_NAVIGATOR = SkillManifest(
    skill_id="jira.task_navigator",
    version="1.0.0",
    description="Jira task navigation: search tasks, fetch issues, and add comments.",
    tool_ids=("jira.search_tasks", "jira.get_issue", "jira.add_comment"),
    prompt_instruction_ids=("jira.task_navigator.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("jira", "tasks", "navigator"),
)

