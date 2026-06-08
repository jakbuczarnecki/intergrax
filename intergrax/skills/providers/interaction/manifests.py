# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

INTERACTION_SESSION_HANDLER = SkillManifest(
    skill_id="interaction.session_handler",
    version="1.0.0",
    description="User session handling: list sessions, read history, and post replies.",
    tool_ids=("interaction.list_sessions", "interaction.get_session_history", "interaction.post_reply"),
    prompt_instruction_ids=("interaction.session_handler.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("interaction", "session", "handler"),
)


INTERACTION_INPUT_CAPTURE = SkillManifest(
    skill_id="interaction.input_capture",
    version="1.0.0",
    description="Capture last user input, post reply, and persist to task memory.",
    tool_ids=("interaction.get_last_input", "interaction.post_reply", "memory.write"),
    prompt_instruction_ids=("interaction.input_capture.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("interaction", "input", "capture"),
)

