# © Artur Czarnecki. All rights reserved.

"""Broken analyzers must not break PredictionEngine or Diagnostic Engine paths."""

from __future__ import annotations

import pytest

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.runtime.prediction import (
    FailurePatternAnalyzer,
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from intergrax.runtime.prediction.prediction_engine import ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_predictive_plugin_failure_isolation() -> None:
    class _Broken(LatencyTrendAnalyzer):
        def analyze(self, context: PredictiveContext) -> tuple:
            raise RuntimeError("plugin boom")

    registry = PredictiveAnalyzerRegistry((_Broken(),))
    engine = PredictionEngine(registry=registry)
    result = engine.analyze(crm_agent_showcase_context())

    assert result.signals == ()
    assert result.audit.degraded is True
    assert any(
        ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE in outcome
        for outcome in result.audit.analyzer_outcomes
    )

    mixed = PredictionEngine(
        registry=PredictiveAnalyzerRegistry(
            (_Broken(), FailurePatternAnalyzer()),
        ),
    )
    partial = mixed.analyze(crm_agent_showcase_context())
    assert partial.signals
    assert partial.audit.degraded is True
