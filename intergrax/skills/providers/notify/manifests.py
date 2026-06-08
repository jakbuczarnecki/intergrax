# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

NOTIFY_SCHEDULED_ALERTS = SkillManifest(
    skill_id="notify.scheduled_alerts",
    version="1.0.0",
    description="Deferred notification scheduling with list, cancel, and immediate send fallback.",
    tool_ids=(
        "notify.schedule",
        "notify.list_scheduled",
        "notify.cancel_scheduled",
        "notify.send",
    ),
    prompt_instruction_ids=("notify.scheduled_alerts.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("notify", "schedule", "alerts"),
)

NOTIFY_BATCH_DISPATCH = SkillManifest(
    skill_id="notify.batch_dispatch",
    version="1.0.0",
    description="Batch notification dispatch with due scheduling and pending list.",
    tool_ids=("notify.send_batch", "notify.dispatch_due", "notify.list_scheduled"),
    prompt_instruction_ids=("notify.batch_dispatch.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("notify", "batch", "dispatch"),
)

