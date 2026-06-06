# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.4–1.9: SignalCollector assembly tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.cost_normalization import normalize_cost_against_budget
from intergrax.runtime.adaptive.signal_collector import (
    SignalAssemblyInput,
    SignalCollector,
    regression_flags_from_signals,
)
from intergrax.runtime.adaptive.signal_store import InMemorySignalStore
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.metrics.export import RunMetricsExport
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.replay.metrics import ExecutionMetrics
from intergrax.runtime.replay.regression import RegressionSignals

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_regression_flags_from_signals_maps_fields() -> None:
    flags = regression_flags_from_signals(
        RegressionSignals(step_explosion=True, llm_cost_spike=True),
    )
    assert "step_explosion" in flags
    assert "llm_cost_spike" in flags


def test_normalize_cost_against_token_budget() -> None:
    ratio = normalize_cost_against_budget(
        total_tokens=5000,
        actual_cost=None,
        run_budget=RunBudget(max_total_tokens=10_000),
    )
    assert ratio == pytest.approx(0.5)


def test_signal_collector_assembles_metrics_and_utility() -> None:
    store = InMemorySignalStore()
    collector = SignalCollector(store, application_id="lab.default")
    metrics = RunMetricsExport(
        run_id="run_1",
        tenant_id="t1",
        agent_id="echo",
        duration_ms=1200,
        event_count=4,
        total_tokens=256,
        cost=0.01,
        behavioral=ExecutionMetrics(
            step_count=3,
            total_llm_calls=2,
            total_tool_calls=1,
            total_artifacts=0,
            total_tokens=256,
            duration=1.2,
            tool_steps_ratio=1 / 3,
            llm_steps_ratio=2 / 3,
        ),
    )
    observation = OnlineEvaluationObservation(
        observation_id="obs_1",
        run_id="run_1",
        agent_id="echo",
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id="harness.default",
        passed=True,
        score=0.9,
    )
    signal = collector.record(
        SignalAssemblyInput(
            run_id="run_1",
            tenant_id="t1",
            application_id="lab.default",
            agent_id="echo",
            task_class="echo.basic",
            run_metrics=metrics,
            regression=RegressionSignals(tool_usage_drop=True),
            evaluation_observation=observation,
            run_budget=RunBudget(max_total_tokens=512),
            hitl_interventions=1,
        )
    )
    assert signal.quality_score == 0.9
    assert signal.eval_mode.value == "shadow"
    assert signal.tool_calls == 1
    assert signal.llm_calls == 2
    assert "tool_usage_drop" in signal.regression_flags
    assert signal.utility is not None
    assert len(store.list_signals()) == 1
