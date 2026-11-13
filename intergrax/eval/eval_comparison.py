# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Advisory baseline-vs-candidate comparison (regression, threshold breach, cost, latency).

The result is an input to governance-owned promotion. This module owns no gate and no
decision: ``EvalComparison.advisory`` is always ``True``.
"""

from __future__ import annotations

from intergrax.contracts.evaluation_contracts import (
    EvalComparison,
    EvalContractError,
    EvalMetricDelta,
    EvalRun,
    EvalRunMetrics,
    EvalTenantBoundaryError,
    EvalThresholdBreach,
    EvalThresholds,
)

_HIGHER_IS_BETTER = {
    "deterministic_pass_rate": True,
    "safety_pass_rate": True,
    "judge_pass_rate": True,
    "human_pass_rate": True,
    "cost_to_success": False,
    "mean_latency_ms": False,
    "p95_latency_ms": False,
}
_RATE_METRICS = frozenset({"deterministic_pass_rate", "safety_pass_rate", "judge_pass_rate", "human_pass_rate"})


def _delta(
    metric: str,
    baseline: float | None,
    candidate: float | None,
    thresholds: EvalThresholds,
) -> EvalMetricDelta:
    higher = _HIGHER_IS_BETTER[metric]
    delta = None if baseline is None or candidate is None else candidate - baseline
    regressed = False
    if baseline is not None:
        if candidate is None:
            regressed = True
        elif higher:
            tolerance = thresholds.rate_regression_tolerance if metric in _RATE_METRICS else 0.0
            regressed = candidate < baseline - tolerance
        else:
            regressed = candidate > baseline * (1.0 + thresholds.cost_latency_regression_ratio)
    return EvalMetricDelta(
        metric=metric,
        higher_is_better=higher,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        regressed=regressed,
    )


def _breaches(metrics: EvalRunMetrics, thresholds: EvalThresholds) -> list[EvalThresholdBreach]:
    out: list[EvalThresholdBreach] = []
    floors = (
        ("deterministic_pass_rate", thresholds.min_deterministic_pass_rate),
        ("safety_pass_rate", thresholds.min_safety_pass_rate),
        ("judge_pass_rate", thresholds.min_judge_pass_rate),
    )
    values = metrics.metric_values()
    for metric, limit in floors:
        if limit is None:
            continue
        observed = values[metric]
        if observed is None or observed < limit:
            out.append(
                EvalThresholdBreach(
                    metric=metric,
                    limit=limit,
                    observed=observed,
                    detail=f"{metric} {observed} below floor {limit}",
                )
            )
    ceilings = (
        ("cost_to_success", thresholds.max_cost_to_success),
        ("mean_latency_ms", thresholds.max_mean_latency_ms),
    )
    for metric, limit in ceilings:
        if limit is None:
            continue
        observed = values[metric]
        if observed is None or observed > limit:
            out.append(
                EvalThresholdBreach(
                    metric=metric,
                    limit=limit,
                    observed=observed,
                    detail=f"{metric} {observed} above ceiling {limit}",
                )
            )
    return out


def compare_eval_runs(
    baseline: EvalRun,
    candidate: EvalRun,
    *,
    thresholds: EvalThresholds | None = None,
) -> EvalComparison:
    """Compare two runs of the same tenant and dataset digest.

    Cross-tenant comparison and comparison across different dataset digests are rejected.
    Thresholds are evaluated on the candidate only.
    """
    if baseline.tenant_id != candidate.tenant_id:
        raise EvalTenantBoundaryError("cannot compare eval runs across tenants")
    if baseline.dataset_digest != candidate.dataset_digest:
        raise EvalContractError("cannot compare eval runs over different dataset digests")
    limits = thresholds or EvalThresholds()

    base_metrics = baseline.metrics()
    cand_metrics = candidate.metrics()
    base_values = base_metrics.metric_values()
    cand_values = cand_metrics.metric_values()
    deltas = tuple(_delta(metric, base_values[metric], cand_values[metric], limits) for metric in _HIGHER_IS_BETTER)

    base_success = baseline.scenario_successes()
    cand_success = candidate.scenario_successes()
    regressions = tuple(sorted(sid for sid, ok in base_success.items() if ok and not cand_success.get(sid, False)))
    improvements = tuple(sorted(sid for sid, ok in cand_success.items() if ok and not base_success.get(sid, False)))

    return EvalComparison(
        tenant_id=baseline.tenant_id,
        dataset_digest=baseline.dataset_digest,
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        baseline_result_digest=baseline.result_digest,
        candidate_result_digest=candidate.result_digest,
        thresholds=limits,
        metric_deltas=deltas,
        scenario_regressions=regressions,
        scenario_improvements=improvements,
        threshold_breaches=tuple(_breaches(cand_metrics, limits)),
        baseline_cost_to_success=base_metrics.cost_to_success,
        candidate_cost_to_success=cand_metrics.cost_to_success,
        baseline_mean_latency_ms=base_metrics.mean_latency_ms,
        candidate_mean_latency_ms=cand_metrics.mean_latency_ms,
    )
