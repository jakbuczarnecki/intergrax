# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Default observation from execution correlation (SELF-HEALING R3)."""

from __future__ import annotations

from datetime import datetime, timezone

from intergrax.contracts.self_healing.execution.context import SelfHealingExecutionContext
from intergrax.contracts.self_healing.observation.provider import ObservationResult


class PlatformExecutionObservationProvider:
    provider_id = "platform.execution.observation"

    def observe(
        self,
        execution_context: SelfHealingExecutionContext,
        *,
        before_metrics: tuple[tuple[str, str], ...] = (),
    ) -> ObservationResult:
        after = (
            ("execution_count", str(len(execution_context.execution_ids))),
            ("attempt_count", str(len(execution_context.operation_attempt_ids))),
        )
        metrics = before_metrics + after
        return ObservationResult(
            metrics=metrics,
            evidence_refs=execution_context.evidence_refs,
            confidence=0.8 if execution_context.execution_ids else 0.3,
            timestamp=datetime.now(timezone.utc),
        )


__all__ = ["PlatformExecutionObservationProvider"]
