# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CODE_RUNNER = SkillManifest(
    skill_id="code.runner",
    version="1.0.0",
    description="Controlled code execution: script run, code exec, and sandbox operation listing.",
    tool_ids=("code.exec", "script.run", "sandbox.list_operations"),
    prompt_instruction_ids=("code.runner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("code", "exec", "script"),
)

