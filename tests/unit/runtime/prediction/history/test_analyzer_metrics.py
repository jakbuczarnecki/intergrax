# © Artur Czarnecki. All rights reserved.

"""Analyzer quality metrics from prediction history."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.predictive_analyzer_quality import compute_analyzer_quality_metrics
from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)
from intergrax.contracts.predictive_risk import PredictiveWindow

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GENERATED = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _history_row(outcome: PredictiveHistoryOutcomeStatus, confidence: float) -> PredictiveRiskHistoryRecord:
    return PredictiveRiskHistoryRecord(
        prediction_run_id="prun_metrics",
        risk_signal_id=f"prsig_{outcome.value}_{confidence}",
        tenant_id="tenant-demo",
        analyzer_id="latency_trend",
        analyzer_version="1.0",
        generated_at=_GENERATED,
        prediction_window=PredictiveWindow(duration_seconds=86400, label="24h"),
        confidence=confidence,
        evidence_refs=("metric:latency",),
        outcome_status=outcome,
        subject_identity="crm_agent",
        risk_type="degradation_risk",
    )


def test_analyzer_metrics() -> None:
    records = tuple(
        _history_row(PredictiveHistoryOutcomeStatus.CONFIRMED, 0.82)
        for _ in range(820)
    ) + tuple(
        _history_row(PredictiveHistoryOutcomeStatus.FALSE_POSITIVE, 0.7)
        for _ in range(100)
    ) + tuple(
        _history_row(PredictiveHistoryOutcomeStatus.UNKNOWN, 0.5)
        for _ in range(80)
    )
    metrics = compute_analyzer_quality_metrics(analyzer_id="latency_trend", records=records)

    assert metrics.prediction_count == 1000
    assert metrics.confirmed_count == 820
    assert metrics.false_positive_count == 100
    assert metrics.unknown_count == 80
    assert abs(metrics.precision - 0.82) < 0.001
