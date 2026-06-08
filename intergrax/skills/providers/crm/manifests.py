# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

CRM_ACCOUNT_LOOKUP = SkillManifest(
    skill_id="crm.account_lookup",
    version="1.0.0",
    description="CRM account research: fetch accounts, list contacts, and browse support tickets.",
    tool_ids=("crm.get_account", "crm.list_contacts", "crm.list_tickets"),
    prompt_instruction_ids=("crm.account_lookup.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("crm", "account", "support"),
)
