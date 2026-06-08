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

SANDBOX_TEST_RUNNER = SkillManifest(
    skill_id="sandbox.test_runner",
    version="1.0.0",
    description="Sandbox test execution with workspace input and error capture.",
    tool_ids=("sandbox.exec", "workspace.read_file", "errors.capture"),
    prompt_instruction_ids=("sandbox.test_runner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("sandbox", "test", "runner"),
)


SANDBOX_REFACTOR_LOOP = SkillManifest(
    skill_id="sandbox.refactor_loop",
    version="1.0.0",
    description="Iterative refactor: exec, write, and workspace search.",
    tool_ids=("sandbox.exec", "workspace.write_file", "workspace.search"),
    prompt_instruction_ids=("sandbox.refactor_loop.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.HIGH,
    tags=("sandbox", "refactor", "loop"),
)

