# © Artur Czarnecki. All rights reserved.

"""PREDICTIVE R1 — predictive incident intelligence qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.contracts.predictive_context import PredictiveContext
from intergrax.contracts.predictive_risk import PredictiveRiskScope
from intergrax.runtime.prediction import (
    FailurePatternAnalyzer,
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from intergrax.runtime.prediction.prediction_engine import ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]
_FORBIDDEN_ROOT_CAUSE_PHRASES = (
    "root cause",
    "is the cause",
    "will fail",
)


def test_prediction_engine_emits_risk_not_problem() -> None:
    registry = PredictiveAnalyzerRegistry(
        (LatencyTrendAnalyzer(), FailurePatternAnalyzer()),
    )
    engine = PredictionEngine(registry=registry)
    result = engine.analyze(crm_agent_showcase_context())

    assert result.signals, "showcase should emit at least one risk signal"
    assert result.audit.prediction_id.startswith("prun_")
    assert result.audit.tenant_id == "tenant-demo"
    for signal in result.signals:
        assert signal.prediction_run_id == result.audit.prediction_id
        assert signal.scope is PredictiveRiskScope.COMPONENT
        assert signal.subject_identity == "crm_agent"
        assert 0.0 < signal.confidence <= 1.0
        assert signal.evidence_refs
        assert signal.analyzer_metadata.analyzer_id
        lowered = signal.summary.lower()
        for phrase in _FORBIDDEN_ROOT_CAUSE_PHRASES:
            assert phrase not in lowered


def test_registry_orders_by_priority() -> None:
    registry = PredictiveAnalyzerRegistry(
        (FailurePatternAnalyzer(), LatencyTrendAnalyzer()),
    )
    ids = [a.analyzer_id for a in registry.analyzers]
    assert ids[0] == "latency_trend"
    assert ids[1] == "failure_pattern"


def test_analyzer_failure_is_contained() -> None:
    class _Broken(LatencyTrendAnalyzer):
        def analyze(self, context: PredictiveContext) -> tuple:
            raise RuntimeError("plugin boom")

    registry = PredictiveAnalyzerRegistry((_Broken(),))
    engine = PredictionEngine(registry=registry)
    result = engine.analyze(crm_agent_showcase_context())
    assert result.signals == ()
    assert result.audit.degraded is True
    assert any(
        ANALYZER_OUTCOME_PLUGIN_UNAVAILABLE in o for o in result.audit.analyzer_outcomes
    )


def test_no_prediction_symbols_in_diagnostic_engine_sources() -> None:
    diag_root = _REPO_ROOT / "intergrax" / "runtime" / "diagnostics"
    forbidden_writes = (
        "PredictiveRiskSignal(",
        "mint_predictive_signal_id",
    )
    for path in diag_root.rglob("*.py"):
        if path.name.startswith("diagnostic_read_service"):
            continue
        text = path.read_text(encoding="utf-8")
        for token in forbidden_writes:
            assert token not in text, f"{path.relative_to(_REPO_ROOT)} must not mint predictions"
