# © Artur Czarnecki. All rights reserved.

"""PDC-D5-R010..R012: baseline vs candidate regression, threshold breach, cost, latency."""

from __future__ import annotations

import pytest
from intergrax.contracts.evaluation_contracts import (
    EvalContractError,
    EvalSubjectOutput,
    EvalTenantBoundaryError,
    EvalThresholds,
)
from intergrax.eval.eval_comparison import compare_eval_runs
from intergrax.eval.evaluators import EvaluatorRegistry, register_existing_evaluators
from tests.qualification.pdc_x.d5._support import (
    ANSWERS,
    JUDGE_MODEL_FOR_TESTS,
    build_run,
    judge_fixed,
    make_dataset,
)

pytestmark = pytest.mark.unit


def _worse_answers() -> dict[str, EvalSubjectOutput]:
    answers = dict(ANSWERS)
    answers["s-math"] = EvalSubjectOutput(output_text="5", cost=1.0, latency_ms=10.0, tokens=10)
    return answers


def test_r010_identical_runs_compare_clean() -> None:
    comparison = compare_eval_runs(build_run(), build_run())
    assert comparison.clean
    assert comparison.advisory is True
    assert comparison.scenario_regressions == ()
    assert all(item.delta in (0.0, None) for item in comparison.metric_deltas)


def test_r010_candidate_regression_is_detected() -> None:
    comparison = compare_eval_runs(build_run(), build_run(answers=_worse_answers()))
    assert comparison.regressed
    assert comparison.scenario_regressions == ("s-math",)
    delta = {item.metric: item for item in comparison.metric_deltas}
    assert delta["deterministic_pass_rate"].regressed
    assert delta["deterministic_pass_rate"].delta is not None
    assert delta["deterministic_pass_rate"].delta < 0


def test_r010_improvement_is_reported_not_flagged() -> None:
    comparison = compare_eval_runs(build_run(answers=_worse_answers()), build_run())
    assert not comparison.regressed
    assert comparison.scenario_improvements == ("s-math",)


def test_r010_comparison_rejects_other_dataset_or_tenant() -> None:
    with pytest.raises(EvalTenantBoundaryError):
        compare_eval_runs(build_run(tenant="tenant-a"), build_run(tenant="tenant-b"))
    with pytest.raises(EvalContractError, match="dataset"):
        compare_eval_runs(build_run(), build_run(dataset=_other_dataset()))


def _other_dataset():
    from intergrax.contracts.evaluation_contracts import EvalDataset

    base = make_dataset()
    return EvalDataset(dataset_id="d5.other", version="1", scenarios=base.scenarios[:2])


def test_r011_threshold_breach_is_flagged_even_without_regression() -> None:
    baseline = build_run(answers=_worse_answers())
    candidate = build_run(answers=_worse_answers())
    comparison = compare_eval_runs(baseline, candidate, thresholds=EvalThresholds(min_deterministic_pass_rate=0.9))
    assert not comparison.regressed
    assert comparison.threshold_breached
    assert comparison.threshold_breaches[0].metric == "deterministic_pass_rate"
    assert not comparison.clean


def test_r011_met_thresholds_are_clean() -> None:
    comparison = compare_eval_runs(
        build_run(),
        build_run(),
        thresholds=EvalThresholds(min_deterministic_pass_rate=0.9, max_cost_to_success=10.0, max_mean_latency_ms=100.0),
    )
    assert comparison.clean


def test_r011_regression_tolerance_is_honoured() -> None:
    worse = build_run(answers=_worse_answers())
    strict = compare_eval_runs(build_run(), worse)
    lenient = compare_eval_runs(
        build_run(),
        worse,
        thresholds=EvalThresholds(rate_regression_tolerance=0.5),
    )
    assert strict.regressed
    # Aggregate tolerance forgives the rate drop, the scenario-level regression stays.
    assert not any(item.regressed for item in lenient.metric_deltas if "rate" in item.metric)
    assert lenient.scenario_regressions == ("s-math",)


def test_r012_comparison_carries_cost_to_success_and_latency() -> None:
    base = build_run()
    answers = {
        key: value.model_copy(update={"cost": value.cost * 2, "latency_ms": value.latency_ms * 3})
        for key, value in ANSWERS.items()
    }
    candidate = build_run(answers=answers)
    comparison = compare_eval_runs(base, candidate)
    assert comparison.baseline_cost_to_success == pytest.approx(2.0)
    assert comparison.candidate_cost_to_success == pytest.approx(4.0)
    assert comparison.baseline_mean_latency_ms == pytest.approx(20.0)
    assert comparison.candidate_mean_latency_ms == pytest.approx(60.0)
    delta = {item.metric: item for item in comparison.metric_deltas}
    assert delta["cost_to_success"].regressed
    assert delta["mean_latency_ms"].regressed
    # Same correctness, so this is a cost/latency regression, not a quality one.
    assert comparison.scenario_regressions == ()


def test_r012_cost_ceiling_breach() -> None:
    comparison = compare_eval_runs(build_run(), build_run(), thresholds=EvalThresholds(max_cost_to_success=1.0))
    assert [b.metric for b in comparison.threshold_breaches] == ["cost_to_success"]


def test_r012_no_successes_means_no_cost_to_success() -> None:
    answers = {key: EvalSubjectOutput(output_text="wrong", cost=1.0, latency_ms=1.0) for key in ANSWERS}
    comparison = compare_eval_runs(build_run(), build_run(answers=answers))
    assert comparison.candidate_cost_to_success is None
    delta = {item.metric: item for item in comparison.metric_deltas}
    assert delta["cost_to_success"].regressed


def test_judge_regression_is_reported_apart_from_rule_regression() -> None:
    def run(score: float):
        return build_run(
            registry=register_existing_evaluators(
                EvaluatorRegistry(),
                judge=("judge.g", judge_fixed(score), JUDGE_MODEL_FOR_TESTS),
            )
        )

    comparison = compare_eval_runs(run(0.9), run(0.1))
    assert comparison.judge_regressed
    assert not comparison.regressed
