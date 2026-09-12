# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy decision input context — no execution handles (SELF-HEALING R6.1)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.audit import AutonomyAuditBundle
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel


@dataclass(frozen=True, slots=True)
class AutonomyDecisionContext:
    """
    Information required to evaluate autonomy posture for one recommendation.

    Contains no executors, commands, or workflow references.
    """

    recommendation_correlation_id: str
    required_autonomy_level: AutonomyLevel
    audit: AutonomyAuditBundle
    operating_context_ref: str | None = None
    knowledge_revision_ref: str | None = None
    governance_assessment_ref: str | None = None

    def __post_init__(self) -> None:
        if not self.recommendation_correlation_id.strip():
            raise ValueError("recommendation_correlation_id required")
        self.required_autonomy_level.ensure_runtime_activatable()
        if self.audit.recommendation_correlation_id != self.recommendation_correlation_id:
            raise ValueError("audit.recommendation_correlation_id mismatch")
        if self.audit.tenant_id.strip() == "":
            raise ValueError("audit.tenant_id required")


__all__ = ["AutonomyDecisionContext"]
