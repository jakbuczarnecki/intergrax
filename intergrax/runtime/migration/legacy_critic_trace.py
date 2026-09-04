# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Historical Critic trace identifiers and diagnostics (DS-MIG-04 retirement evidence)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

LEGACY_CRITIC_STEP_L0_FAILED = "critic.l0_failed"
LEGACY_CRITIC_STEP_L1_JUDGE = "critic.l1_judge"
LEGACY_CRITIC_STEP_TRAJECTORY = "critic.trajectory"
LEGACY_CRITIC_STEP_EVALUATOR_LOOP = "critic.evaluator_loop"
LEGACY_CRITIC_STEP_FINAL_VERDICT = "critic.final_verdict"


class LegacyCriticVerdictDiagV1(BaseModel):
    """Persisted critic verdict diagnostic payload (historical traces only)."""

    model_config = ConfigDict(extra="forbid")

    scope: str
    passed: bool
    recommended_action: str
    layer: str
    score: float | None = None
    failure_reasons: tuple[str, ...] = ()
    agent_id: str = ""
    node_id: str | None = None
