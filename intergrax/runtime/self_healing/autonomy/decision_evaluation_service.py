# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Autonomy decision evaluation service — explainable evaluation only (SELF-HEALING R6.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from intergrax.contracts.self_healing.autonomy.decision_evaluator import AutonomyDecisionEvaluator
from intergrax.contracts.self_healing.autonomy.evaluation_audit import (
    AutonomyEvaluationAuditRecorder,
    AutonomyEvaluationAuditTrailEntry,
)
from intergrax.contracts.self_healing.autonomy.evaluation_repository import AutonomyDecisionRepository
from intergrax.contracts.self_healing.autonomy.evaluation_result import AutonomyEvaluationResult
from intergrax.contracts.self_healing.autonomy.request import AutonomyControlRequest


@dataclass(frozen=True, slots=True)
class AutonomyDecisionEvaluationService:
    evaluator: AutonomyDecisionEvaluator
    repository: AutonomyDecisionRepository | None = None
    audit_recorder: AutonomyEvaluationAuditRecorder | None = None

    def evaluate(self, request: AutonomyControlRequest) -> AutonomyEvaluationResult:
        result = self.evaluator.evaluate(request)
        if self.audit_recorder is not None:
            entry = AutonomyEvaluationAuditTrailEntry.from_evaluation_result(
                result,
                recorded_at=datetime.now(tz=timezone.utc),
            )
            self.audit_recorder.record(entry)
        if self.repository is not None:
            return self.repository.append_evaluation(result)
        return result


__all__ = ["AutonomyDecisionEvaluationService"]
