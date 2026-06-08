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

MESSAGE_BUS_TASK_ADMIN = SkillManifest(
    skill_id="message_bus.task_admin",
    version="1.0.0",
    description="Message bus task administration: list, cancel, and purge completed tasks.",
    tool_ids=("message_bus.list_tasks", "message_bus.cancel", "message_bus.purge_completed"),
    prompt_instruction_ids=("message_bus.task_admin.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("message_bus", "admin", "tasks"),
)

MESSAGE_BUS_RETRY_HANDLER = SkillManifest(
    skill_id="message_bus.retry_handler",
    version="1.0.0",
    description="Retry failed async tasks via re-enqueue and status poll.",
    tool_ids=("message_bus.get_status", "message_bus.enqueue", "notify.send"),
    prompt_instruction_ids=("message_bus.retry_handler.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("message_bus", "retry", "handler"),
)


MESSAGE_BUS_DEAD_LETTER = SkillManifest(
    skill_id="message_bus.dead_letter",
    version="1.0.0",
    description="Dead-letter hygiene: list, purge completed, and log search.",
    tool_ids=("message_bus.list_tasks", "message_bus.purge_completed", "logs.search"),
    prompt_instruction_ids=("message_bus.dead_letter.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("message_bus", "dead_letter", "hygiene"),
)

