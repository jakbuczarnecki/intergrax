# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

COST_BUDGET_GUARDIAN = SkillManifest(
    skill_id="cost.budget_guardian",
    version="1.0.0",
    description="Run budget enforcement: check quotas, forecast spend, and gate expensive agent actions.",
    tool_ids=("cost.check_quota", "cost.get_run_budget", "cost.forecast_spend"),
    prompt_instruction_ids=("cost.budget_guardian.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("cost", "budget", "governance"),
)

COST_CHARGEBACK_REPORT = SkillManifest(
    skill_id="cost.chargeback_report",
    version="1.0.0",
    description="Chargeback report from run budget, billing usage, and workspace export.",
    tool_ids=("cost.get_run_budget", "billing.list_usage", "workspace.write_file"),
    prompt_instruction_ids=("cost.chargeback_report.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("cost", "chargeback", "report"),
)

