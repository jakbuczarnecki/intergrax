# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

BILLING_USAGE_TRACKER = SkillManifest(
    skill_id="billing.usage_tracker",
    version="1.0.0",
    description="Usage metering: list billing records, record usage events, and correlate run costs.",
    tool_ids=("billing.list_usage", "billing.record_usage", "harness.get_run_cost"),
    prompt_instruction_ids=("billing.usage_tracker.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("billing", "usage", "metering"),
)
