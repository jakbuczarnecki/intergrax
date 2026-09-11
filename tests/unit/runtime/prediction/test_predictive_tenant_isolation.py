# © Artur Czarnecki. All rights reserved.

"""Tenant-scoped evidence must not produce cross-tenant risk signals."""

from __future__ import annotations

import pytest

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.runtime.prediction import LatencyTrendAnalyzer, PredictionEngine, PredictiveAnalyzerRegistry
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_predictive_tenant_isolation() -> None:
    engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)))

    tenant_a = engine.analyze(crm_agent_showcase_context(tenant_id="tenant-a"))
    tenant_b = engine.analyze(crm_agent_showcase_context(tenant_id="tenant-b"))

    for signal in tenant_a.signals:
        assert signal.tenant_id == "tenant-a"
    for signal in tenant_b.signals:
        assert signal.tenant_id == "tenant-b"

    class _CrossTenantEmit:
        analyzer_id = "cross_tenant_emit"
        analyzer_namespace = "test"
        priority = 1
        model_version = "cross@1"

        def analyze(self, context: PredictiveContext) -> tuple:
            from datetime import UTC, datetime

            from intergrax.contracts.predictive_risk import (
                PREDICTION_RUN_ID_ENGINE_STAMP,
                PredictiveAnalyzerMetadata,
                PredictiveRiskScope,
                PredictiveRiskSeverity,
                PredictiveRiskSignal,
                PredictiveWindow,
                mint_predictive_signal_id,
            )

            return (
                PredictiveRiskSignal(
                    signal_id=mint_predictive_signal_id(),
                    prediction_run_id=PREDICTION_RUN_ID_ENGINE_STAMP,
                    tenant_id="tenant-evil",
                    scope=PredictiveRiskScope.COMPONENT,
                    subject_identity="crm_agent",
                    risk_type="test_cross",
                    severity=PredictiveRiskSeverity.LOW,
                    confidence=0.5,
                    evidence_refs=("test:cross_tenant",),
                    prediction_window=PredictiveWindow(duration_seconds=60, label="1m"),
                    generated_at=datetime.now(tz=UTC),
                    analyzer_metadata=PredictiveAnalyzerMetadata(
                        analyzer_id="cross_tenant_emit",
                        analyzer_version="cross@1",
                    ),
                    model_version="cross@1",
                    summary="cross tenant probe",
                ),
            )

    cross_engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((_CrossTenantEmit(),)))
    with pytest.raises(ValueError, match="cross-tenant"):
        cross_engine.analyze(crm_agent_showcase_context(tenant_id="tenant-a"))
