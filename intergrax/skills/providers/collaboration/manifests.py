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

COLLABORATION_THREAD_REPLY = SkillManifest(
    skill_id="collaboration.thread_reply",
    version="1.0.0",
    description="Email thread follow-up: read messages, list threads, and send replies.",
    tool_ids=(
        "collaboration.reply_message",
        "collaboration.get_message",
        "collaboration.list_messages",
    ),
    prompt_instruction_ids=("collaboration.thread_reply.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("collaboration", "email", "reply"),
)

COLLABORATION_MEETING_BRIEF = SkillManifest(
    skill_id="collaboration.meeting_brief",
    version="1.0.0",
    description="Meeting brief from calendar, user profile, and workspace draft.",
    tool_ids=("collaboration.list_calendar", "collaboration.get_user", "workspace.write_file"),
    prompt_instruction_ids=("collaboration.meeting_brief.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("collaboration", "meeting", "brief"),
)


COLLABORATION_STAKEHOLDER_PING = SkillManifest(
    skill_id="collaboration.stakeholder_ping",
    version="1.0.0",
    description="Stakeholder outreach with CRM context, mail, and notify.",
    tool_ids=("crm.get_account", "collaboration.send_mail", "notify.send"),
    prompt_instruction_ids=("collaboration.stakeholder_ping.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("collaboration", "stakeholder", "ping"),
)

