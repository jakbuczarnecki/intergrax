# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

WORKSPACE_AUTHORING = SkillManifest(
    skill_id="workspace.authoring",
    version="1.0.0",
    description="Shadow workspace read/write/search with task memory persistence for drafts.",
    tool_ids=("workspace.read_file", "workspace.write_file", "workspace.search", "memory.write"),
    prompt_instruction_ids=("workspace.authoring.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("workspace", "authoring", "shadow"),
)
