# © Artur Czarnecki. All rights reserved.

"""Predictive layer must not mint Problems or diagnostic authority."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.runtime.prediction import (
    FailurePatternAnalyzer,
    LatencyTrendAnalyzer,
    PredictionEngine,
    PredictiveAnalyzerRegistry,
)
from tests.unit.runtime.prediction.conftest import crm_agent_showcase_context

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]


def test_predictive_authority_boundary() -> None:
    engine = PredictionEngine(
        registry=PredictiveAnalyzerRegistry(
            (LatencyTrendAnalyzer(), FailurePatternAnalyzer()),
        ),
    )
    result = engine.analyze(crm_agent_showcase_context())
    assert result.signals
    assert not hasattr(result, "problem_id")
    assert not hasattr(result.audit, "problem_occurrence_id")

    prediction_root = _REPO_ROOT / "intergrax" / "runtime" / "prediction"
    forbidden = (
        "ProblemLifecycleEngine",
        "mint_problem",
        "create_problem",
        "ProblemPersistence",
    )
    for path in prediction_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{path.name} must not own Problem authority ({token})"
