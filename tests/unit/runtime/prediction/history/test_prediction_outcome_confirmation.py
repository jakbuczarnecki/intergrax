# © Artur Czarnecki. All rights reserved.

"""Prediction plus future evidence resolves to CONFIRMED."""

from __future__ import annotations

import pytest

from intergrax.contracts.predictive_history import PredictiveHistoryOutcomeStatus
from intergrax.runtime.prediction import (
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from intergrax.runtime.prediction.history import (
    InMemoryPredictiveHistoryPersistence,
    PredictiveHistoryService,
)
from tests.unit.runtime.prediction.conftest import (
    crm_agent_day2_future_evidence,
    crm_agent_showcase_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_prediction_outcome_confirmation() -> None:
    engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)))
    prediction = engine.analyze(crm_agent_showcase_context())
    assert prediction.signals

    service = PredictiveHistoryService(persistence=InMemoryPredictiveHistoryPersistence())
    stored = service.record_signals(prediction.signals)
    evaluated = service.resolve_and_persist(stored[0], crm_agent_day2_future_evidence())

    assert evaluated.outcome_status is PredictiveHistoryOutcomeStatus.EVALUATED
    prior = service.persistence.get(
        tenant_id=evaluated.tenant_id,
        risk_signal_id=evaluated.risk_signal_id,
    )
    assert prior is not None
    assert prior.outcome_status is PredictiveHistoryOutcomeStatus.EVALUATED

    resolver_only = service.outcome_resolver.resolve(stored[0], crm_agent_day2_future_evidence())
    assert resolver_only.outcome_status is PredictiveHistoryOutcomeStatus.CONFIRMED
