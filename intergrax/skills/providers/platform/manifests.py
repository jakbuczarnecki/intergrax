# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

PLATFORM_CONCIERGE = SkillManifest(
    skill_id="platform.concierge",
    version="1.0.0",
    description=(
        "Intergrax assistant hub: retrieval, web evidence, session memory, and skill introspection."
    ),
    tool_ids=("rag.retrieve", "websearch.query", "memory.read", "skill.resolve"),
    prompt_instruction_ids=("platform.concierge.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("platform", "concierge", "assistant"),
)

PLATFORM_SECRETS_FLAGS = SkillManifest(
    skill_id="platform.secrets_flags",
    version="1.0.0",
    description="Runtime secret fetch and feature-flag evaluation for governed platform agents.",
    tool_ids=("platform.get_secret", "platform.evaluate_feature_flag"),
    prompt_instruction_ids=("platform.secrets_flags.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("platform", "secrets", "feature_flags"),
)

PLATFORM_CICD_INSPECTOR = SkillManifest(
    skill_id="platform.cicd_inspector",
    version="1.0.0",
    description="CI/CD inspection: list workflow runs, fetch run details, and list check suites.",
    tool_ids=(
        "platform.list_workflow_runs",
        "platform.get_workflow_run",
        "platform.list_check_suites",
    ),
    prompt_instruction_ids=("platform.cicd_inspector.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("platform", "cicd", "inspector"),
)
