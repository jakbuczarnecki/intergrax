# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

EVAL_SCORE_LOGGER = SkillManifest(
    skill_id="eval.score_logger",
    version="1.0.0",
    description="Evaluation harness: log scores to Braintrust and correlate with trace queries.",
    tool_ids=("braintrust.log_eval", "observability.query_traces"),
    prompt_instruction_ids=("eval.score_logger.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("eval", "braintrust", "observability"),
)
