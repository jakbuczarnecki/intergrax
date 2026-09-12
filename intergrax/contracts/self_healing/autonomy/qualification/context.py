# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Read-only snapshot for autonomy safety qualification (SELF-HEALING R6.4)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.contracts.self_healing.autonomy.evaluation_audit import AutonomyEvaluationAuditRecorder
from intergrax.contracts.self_healing.autonomy.execution_audit import AutonomyExecutionAuditRepository
from intergrax.contracts.self_healing.autonomy.execution_boundary import AutonomyExecutionBoundary
from intergrax.contracts.self_healing.autonomy.guard import AutonomyExecutionGuard
from intergrax.contracts.self_healing.autonomy.level import AutonomyLevel


@dataclass(frozen=True, slots=True)
class AutonomySafetyQualificationContext:
    """
    Configuration snapshot for qualification — no executor handles or runtime mutation.

    ``direct_executor_access_enabled`` must be set when hosts declare a bypass of the guard/boundary.
    """

    tenant_id: str
    default_autonomy_level: AutonomyLevel
    execution_guard: AutonomyExecutionGuard | None = None
    execution_boundary: AutonomyExecutionBoundary | None = None
    evaluation_audit_recorder: AutonomyEvaluationAuditRecorder | None = None
    execution_audit_repository: AutonomyExecutionAuditRepository | None = None
    direct_executor_access_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.tenant_id.strip():
            raise ValueError("tenant_id required")


__all__ = ["AutonomySafetyQualificationContext"]
