# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

COLLABORATION_OUTREACH = SkillManifest(
    skill_id="collaboration.outreach",
    version="1.0.0",
    description="Email outreach: list threads, read messages, and send mail via collaboration suite.",
    tool_ids=(
        "collaboration.send_mail",
        "collaboration.list_messages",
        "collaboration.get_message",
    ),
    prompt_instruction_ids=("collaboration.outreach.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("collaboration", "email", "outreach"),
)

COLLABORATION_CALENDAR = SkillManifest(
    skill_id="collaboration.calendar",
    version="1.0.0",
    description="Calendar scheduling: list events, create meetings, and resolve user profiles.",
    tool_ids=(
        "collaboration.list_calendar",
        "collaboration.create_event",
        "collaboration.get_user",
    ),
    prompt_instruction_ids=("collaboration.calendar.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("collaboration", "calendar", "scheduling"),
)
