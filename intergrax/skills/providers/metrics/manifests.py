# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

METRICS_RUN_OBSERVER = SkillManifest(
    skill_id="metrics.run_observer",
    version="1.0.0",
    description="Runtime metrics observer: instant/range queries correlated with trace lookup.",
    tool_ids=(
        "metrics.query_instant",
        "metrics.query_range",
        "observability.query_traces",
    ),
    prompt_instruction_ids=("metrics.run_observer.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("metrics", "observability", "monitoring"),
)
