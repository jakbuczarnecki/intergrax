# © Artur Czarnecki. All rights reserved.

"""Tenant A prediction history must not leak into tenant B."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from intergrax.contracts.predictive_history import (
    PredictiveHistoryOutcomeStatus,
    PredictiveRiskHistoryRecord,
)
from intergrax.contracts.predictive_risk import PredictiveWindow
from intergrax.runtime.prediction.history import InMemoryPredictiveHistoryPersistence

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_GENERATED = datetime(2026, 9, 11, 12, 0, tzinfo=UTC)


def _record(tenant_id: str, signal_id: str) -> PredictiveRiskHistoryRecord:
    return PredictiveRiskHistoryRecord(
        prediction_run_id=f"prun_{tenant_id}",
        risk_signal_id=signal_id,
        tenant_id=tenant_id,
        analyzer_id="latency_trend",
        analyzer_version="1.0",
        generated_at=_GENERATED,
        prediction_window=PredictiveWindow(duration_seconds=86400, label="24h"),
        confidence=0.8,
        evidence_refs=(f"tenant:{tenant_id}",),
        outcome_status=PredictiveHistoryOutcomeStatus.CREATED,
        subject_identity="crm_agent",
        risk_type="degradation_risk",
    )


def test_prediction_tenant_isolation() -> None:
    store = InMemoryPredictiveHistoryPersistence()
    store.append(_record("tenant-a", "prsig_a"))
    store.append(_record("tenant-b", "prsig_b"))

    list_a = store.list_for_tenant(tenant_id="tenant-a")
    list_b = store.list_for_tenant(tenant_id="tenant-b")

    assert len(list_a) == 1
    assert len(list_b) == 1
    assert list_a[0].tenant_id == "tenant-a"
    assert list_b[0].tenant_id == "tenant-b"
    assert store.get(tenant_id="tenant-a", risk_signal_id="prsig_b") is None
