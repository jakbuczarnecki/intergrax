# © Artur Czarnecki. All rights reserved.

from intergrax.skills.core.contracts import SkillManifest, SkillRiskTier

ML_EXPLAIN_PREDICT = SkillManifest(
    skill_id="ml.explain_predict",
    version="1.0.0",
    description="ML inference with explainability: predict, explain, and batch predict.",
    tool_ids=("ml.predict", "ml.explain", "ml.batch_predict"),
    prompt_instruction_ids=("ml.explain_predict.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("ml", "predict", "explain"),
)

