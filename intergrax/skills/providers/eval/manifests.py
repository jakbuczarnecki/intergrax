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

EVAL_TRAJECTORY_JUDGE = SkillManifest(
    skill_id="eval.trajectory_judge",
    version="1.0.0",
    description="Trajectory evaluation: judge agent runs, record observations, and inspect trajectories.",
    tool_ids=("eval.judge", "eval.record_observation", "eval.trajectory"),
    prompt_instruction_ids=("eval.trajectory_judge.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.MEDIUM,
    tags=("eval", "trajectory", "judge"),
)

EVAL_RELEASE_COMPARE = SkillManifest(
    skill_id="eval.release_compare",
    version="1.0.0",
    description="Release comparison: compare eval releases, summarize deltas, and export observations.",
    tool_ids=("eval.compare_releases", "eval.summarize_release", "eval.export_observations"),
    prompt_instruction_ids=("eval.release_compare.system",),
    policy_fragment_id=None,
    risk_tier=SkillRiskTier.LOW,
    tags=("eval", "release", "compare"),
)
