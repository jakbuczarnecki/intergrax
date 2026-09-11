# © Artur Czarnecki. All rights reserved.

"""Prediction history must not become Problem or incident authority."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.prediction.history import (
    InMemoryPredictiveHistoryPersistence,
    PredictiveHistoryService,
)
from intergrax.runtime.prediction import PredictionEngine, PredictiveAnalyzerRegistry, LatencyTrendAnalyzer
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[5]


def test_prediction_history_not_problem_store() -> None:
    engine = PredictionEngine(registry=PredictiveAnalyzerRegistry((LatencyTrendAnalyzer(),)))
    result = engine.analyze(crm_agent_showcase_context())
    history = PredictiveHistoryService(persistence=InMemoryPredictiveHistoryPersistence())
    records = history.record_signals(result.signals)
    assert records
    assert not hasattr(records[0], "problem_id")

    roots = (
        _REPO_ROOT / "intergrax" / "runtime" / "prediction" / "history",
        _REPO_ROOT / "intergrax" / "runtime" / "prediction" / "outcome",
    )
    forbidden = (
        "ProblemLifecycleEngine",
        "mint_problem",
        "create_problem",
        "ProblemPersistence",
        "DiagnosticProblem",
    )
    for root in roots:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for token in forbidden:
                assert token not in text, f"{path.name} must not own Problem authority ({token})"
