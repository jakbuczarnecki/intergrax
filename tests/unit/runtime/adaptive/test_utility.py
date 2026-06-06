# © Artur Czarnecki. All rights reserved.

"""W-ADAPT-1.8: Utility function tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.adaptive.contracts import UtilityWeights
from intergrax.runtime.adaptive.utility import compute_utility, regression_penalty

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def test_compute_utility_quality_dominates_by_default() -> None:
    high = compute_utility(
        quality_score=1.0,
        cost_normalized=0.5,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=[],
    )
    low = compute_utility(
        quality_score=0.2,
        cost_normalized=0.5,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=[],
    )
    assert high > low


def test_compute_utility_cost_over_budget_penalized() -> None:
    at_budget = compute_utility(
        quality_score=0.8,
        cost_normalized=1.0,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=[],
    )
    over_budget = compute_utility(
        quality_score=0.8,
        cost_normalized=1.5,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=[],
    )
    assert at_budget > over_budget


def test_compute_utility_regression_flags_reduce_score() -> None:
    clean = compute_utility(
        quality_score=0.8,
        cost_normalized=1.0,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=[],
    )
    regressed = compute_utility(
        quality_score=0.8,
        cost_normalized=1.0,
        latency_ms=1000,
        hitl_interventions=0,
        regression_flags=["step_explosion", "llm_cost_spike"],
    )
    assert clean > regressed


def test_regression_penalty_is_bounded() -> None:
    assert regression_penalty([]) == 0.0
    assert regression_penalty(["a", "b", "c", "d", "e"]) == 1.0


def test_custom_weights_respected() -> None:
    weights = UtilityWeights(w_quality=0.0, w_cost=1.0, w_latency=0.0, w_hitl=0.0, w_regression=0.0)
    value = compute_utility(
        quality_score=1.0,
        cost_normalized=2.0,
        latency_ms=0,
        hitl_interventions=0,
        regression_flags=[],
        weights=weights,
    )
    assert value == pytest.approx(-1.0)
