# © Artur Czarnecki. All rights reserved.

"""Evaluator-loop configuration for critique-revise cycles (Phase CRIT-V-1.3)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluatorLoopSpec(BaseModel):
    """
    Bounded critique-revise loop for ``CoordinationPattern.EVALUATOR_LOOP``.

    When validation fails and iterations remain, the graph routes to ``revise_node_id``.
    """

    model_config = ConfigDict(extra="forbid")

    max_iterations: int = Field(default=2, ge=1, le=16)
    min_score: float = Field(default=0.75, ge=0.0, le=1.0)
    revise_node_id: str | None = None
    escalate_on_exhaustion: bool = True

    @model_validator(mode="after")
    def _validate_revise_target(self) -> EvaluatorLoopSpec:
        if self.max_iterations > 1 and not (self.revise_node_id or "").strip():
            raise ValueError("revise_node_id is required when max_iterations > 1")
        return self
