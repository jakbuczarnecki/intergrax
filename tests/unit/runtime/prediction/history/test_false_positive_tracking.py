# © Artur Czarnecki. All rights reserved.

"""Wrong predictions classify as false positive without incident creation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)
from intergrax.contracts.predictive_risk import PredictiveWindow
from intergrax.runtime.diagnostics.in_memory_problem_persistence import InMemoryProblemPersistence
from intergrax.runtime.prediction.history import InMemoryPredictiveHistoryPersistence
from intergrax.runtime.prediction.outcome import (
    PredictionFutureEvidenceSnapshot,
    PredictionOutcomeResolver,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GENERATED = datetime(2026, 9, 10, 12, 0, tzinfo=UTC)


def test_false_positive_tracking() -> None:
    record = PredictiveRiskHistoryRecord(
        prediction_run_id="prun_test",
        risk_signal_id="prsig_test",
        tenant_id="tenant-demo",
        analyzer_id="latency_trend",
        analyzer_version="1.0",
        generated_at=_GENERATED,
        prediction_window=PredictiveWindow(duration_seconds=3600, label="1h"),
        confidence=0.86,
        evidence_refs=("metric:billing_latency",),
        outcome_status=PredictiveHistoryOutcomeStatus.OBSERVED,
        subject_identity="billing_agent",
        risk_type="degradation_risk",
        summary="Billing degradation risk",
    )
    future = PredictionFutureEvidenceSnapshot(
        observed_at=_GENERATED + timedelta(hours=2),
        evidence_refs=("metric:crm_latency",),
        execution_failed=False,
        problem_created_for_subject=False,
        matching_risk_keywords=("crm_agent",),
    )
    resolution = PredictionOutcomeResolver().resolve(record, future)
    assert resolution.outcome_status is PredictiveHistoryOutcomeStatus.FALSE_POSITIVE

    problems = InMemoryProblemPersistence()
    assert problems.query_problems(tenant_id="tenant-demo", limit=10).problems == ()
