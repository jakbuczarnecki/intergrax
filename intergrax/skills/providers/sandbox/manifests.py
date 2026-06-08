# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

SANDBOX_CODE_EXEC = SkillManifest(
    skill_id="sandbox.code_exec",
    version="1.0.0",
    description="Sandboxed code execution with shadow workspace read/write for agent coding tasks.",
    tool_ids=("sandbox.exec", "workspace.read_file", "workspace.write_file"),
    prompt_instruction_ids=("sandbox.code_exec.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("sandbox", "code", "exec"),
)
