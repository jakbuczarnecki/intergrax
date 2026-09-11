# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Bounded predictive analysis orchestration — risk signals only (PREDICTIVE R1)."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from intergrax.contracts.predictive import PredictiveContext, PredictiveScope
from intergrax.contracts.predictive.audit import PredictionAuditRecord
from intergrax.contracts.predictive.quality import PredictiveQualityAssessment
from intergrax.contracts.predictive.snapshot import PredictiveContextSnapshot
from intergrax.contracts.predictive_risk import (
    PredictiveRiskSignal,
    mint_prediction_run_id,
)
from intergrax.runtime.prediction.context.predictive_context_builder import (
    PredictiveContextBuilder,
)
from intergrax.runtime.prediction.governance.analyzer_quality_store import (
    InMemoryPredictiveAnalyzerQualityStore,
)
from intergrax.runtime.prediction.governance.prediction_governance_layer import (
    PredictionGovernanceLayer,
)
from intergrax.runtime.prediction.predictive_registry import PredictiveAnalyzerRegistry

DEFAULT_PREDICTION_TIME_BUDGET_MS = 500
ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class PredictionEngineResult:
    signals: tuple[PredictiveRiskSignal, ...]
    audit: PredictionAuditRecord
    quality_by_signal_id: dict[str, PredictiveQualityAssessment]
    context_snapshot: PredictiveContextSnapshot


@dataclass(slots=True)
class PredictionEngine:
    """Run registered analyzers with timeout and failure containment."""

    registry: PredictiveAnalyzerRegistry
    time_budget_ms: int = DEFAULT_PREDICTION_TIME_BUDGET_MS
    context_builder: PredictiveContextBuilder | None = None
    governance: PredictionGovernanceLayer | None = None

    def __post_init__(self) -> None:
        if self.governance is None:
            self.governance = PredictionGovernanceLayer(
                quality_store=InMemoryPredictiveAnalyzerQualityStore(),
            )

    def analyze(self, context: PredictiveContext) -> PredictionEngineResult:
        if context.tenant_id.strip() == "":
            raise ValueError("context.tenant_id required")

        prediction_run_id = mint_prediction_run_id()
        deadline = time.monotonic() + (self.time_budget_ms / 1000.0)
        signals: list[PredictiveRiskSignal] = []
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
            outcomes.append(f"{analyzer.analyzer_id}:ok:{len(batch)}")

        assert self.governance is not None
        governed = self.governance.govern_run(
            context=context,
            prediction_run_id=prediction_run_id,
            raw_signals=tuple(signals),
            analyzer_outcomes=tuple(outcomes),
            degraded=degraded,
            generated_at=generated_at,
        )
        return PredictionEngineResult(
            signals=governed.signals,
            audit=governed.audit,
            quality_by_signal_id=governed.quality_by_signal_id,
            context_snapshot=governed.context_snapshot,
        )

    def analyze_scope(self, scope: PredictiveScope) -> PredictionEngineResult:
        if self.context_builder is None:
            raise ValueError("context_builder required for analyze_scope")
        context = self.context_builder.build(scope)
        return self.analyze(context)


__all__ = [
    "ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE",
    "DEFAULT_PREDICTION_TIME_BUDGET_MS",
    "PredictionAuditRecord",
    "PredictionEngine",
    "PredictionEngineResult",
]
