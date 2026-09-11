# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded predictive analysis orchestration — risk signals only (PREDICTIVE R1)."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_risk import (
    PredictiveRiskSignal,
    mint_prediction_run_id,
)
from intergrax.runtime.prediction.predictive_registry import PredictiveAnalyzerRegistry

DEFAULT_PREDICTION_TIME_BUDGET_MS = 500
ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PredictionAuditRecord:
    """Audit envelope for one prediction run — not diagnostic authority."""

    prediction_id: str
    tenant_id: str
    input_snapshot_id: str
    generated_at: datetime
    model_versions: tuple[str, ...]
    signal_ids: tuple[str, ...]
    analyzer_outcomes: tuple[str, ...]
    degraded: bool


@dataclass(frozen=True, slots=True)
class PredictionEngineResult:
    signals: tuple[PredictiveRiskSignal, ...]
    audit: PredictionAuditRecord


@dataclass(slots=True)
class PredictionEngine:
    """Run registered analyzers with timeout and failure containment."""

    registry: PredictiveAnalyzerRegistry
    time_budget_ms: int = DEFAULT_PREDICTION_TIME_BUDGET_MS

    def analyze(self, context: PredictiveContext) -> PredictionEngineResult:
        if context.tenant_id.strip() == "":
            raise ValueError("context.tenant_id required")

        prediction_run_id = mint_prediction_run_id()
        deadline = time.monotonic() + (self.time_budget_ms / 1000.0)
        signals: list[PredictiveRiskSignal] = []
        model_versions: list[str] = []
        outcomes: list[str] = []
        degraded = False
        generated_at = datetime.now(tz=UTC)

        for analyzer in self.registry.analyzers:
            if time.monotonic() > deadline:
                degraded = True
                outcomes.append(f"{analyzer.analyzer_id}:skipped_time_budget")
                break
            try:
                batch = analyzer.analyze(context)
            except Exception:
                degraded = True
                outcomes.append(f"{analyzer.analyzer_id}:{ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE}")
                continue
            stamped: list[PredictiveRiskSignal] = []
            for signal in batch:
                if signal.tenant_id != context.tenant_id:
                    raise ValueError("analyzer emitted cross-tenant risk signal")
                stamped.append(
                    replace(
                        signal,
                        prediction_run_id=prediction_run_id,
                    ),
                )
            signals.extend(stamped)
            model_versions.append(analyzer.model_version)
            outcomes.append(f"{analyzer.analyzer_id}:ok:{len(batch)}")

        audit = PredictionAuditRecord(
            prediction_id=prediction_run_id,
            tenant_id=context.tenant_id,
            input_snapshot_id=context.input_snapshot_id,
            generated_at=generated_at,
            model_versions=tuple(dict.fromkeys(model_versions)),
            signal_ids=tuple(s.signal_id for s in signals),
            analyzer_outcomes=tuple(outcomes),
            degraded=degraded,
        )
        return PredictionEngineResult(signals=tuple(signals), audit=audit)


__all__ = [
    "ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE",
    "DEFAULT_PREDICTION_TIME_BUDGET_MS",
    "PredictionAuditRecord",
    "PredictionEngine",
    "PredictionEngineResult",
]
