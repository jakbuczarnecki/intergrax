# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CODECRAFT_EPHEMERAL_BUILDER = SkillManifest(
    skill_id="codecraft.ephemeral_builder",
    version="1.0.0",
    description="Harness ephemeral code craft loop: start, iterate, promote, dispose.",
    tool_ids=(
        "codecraft.start",
        "codecraft.iterate",
        "codecraft.get_state",
        "codecraft.promote",
        "codecraft.dispose",
        "codecraft.list_ephemeral_tools",
        "workspace.read_file",
        "workspace.write_file",
    ),
    prompt_instruction_ids=("codecraft.ephemeral_builder.system",),
    policy_fragment_id="codecraft_governance",
    risk_tier=SkillRiskTier.HIGH,
    tags=("codecraft", "ephemeral", "builder"),
)
