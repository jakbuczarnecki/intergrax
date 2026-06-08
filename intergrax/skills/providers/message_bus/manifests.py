# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

MESSAGE_BUS_ASYNC_RUNNER = SkillManifest(
    skill_id="message_bus.async_runner",
    version="1.0.0",
    description="Async task queue: enqueue work, poll status, and fetch results via message bus.",
    tool_ids=("message_bus.enqueue", "message_bus.get_status", "message_bus.get_result"),
    prompt_instruction_ids=("message_bus.async_runner.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("message_bus", "async", "queue"),
)
