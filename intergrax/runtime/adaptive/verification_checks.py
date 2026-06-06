# © Artur Czarnecki. All rights reserved.

"""Verification check implementations reusing harness architecture modules (W-ADAPT-5.1–5.5)."""

from __future__ import annotations

from typing import Protocol

from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal, OutcomeEvalMode
from intergrax.runtime.adaptive.verification_models import (
    VerificationCheckId,
    VerificationCheckResult,
    VerificationContext,
)
from intergrax.runtime.architecture.cost_budget import BudgetEnvelope, evaluate_budget_envelopes
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationRegistryTrendReport,
)


class SecurityAdversarialBaselineChecker(Protocol):
    """Protocol for V-SEC adversarial baseline evaluation."""

    def evaluate(self) -> bool:
        """Return True when prompt/tool/retrieval adversarial suites are green."""
        ...


class HarnessSecurityAdversarialBaselineChecker:
    """Default checker delegating to maturity gate security harness baseline."""

    def evaluate(self) -> bool:
        from intergrax.runtime.architecture.maturity_gate_evidence import (
            evaluate_security_adversarial_baseline,
        )

        return evaluate_security_adversarial_baseline()


def split_candidate_baseline_signals(
    signals: list[HarnessOutcomeSignal],
) -> tuple[list[HarnessOutcomeSignal], list[HarnessOutcomeSignal]]:
    """Shadow-mode signals are candidate runs; others form the baseline window."""
    candidate = [item for item in signals if item.eval_mode == OutcomeEvalMode.SHADOW]
    baseline = [item for item in signals if item.eval_mode != OutcomeEvalMode.SHADOW]
    return candidate, baseline


def mean_utility(signals: list[HarnessOutcomeSignal]) -> float | None:
    utilities = [item.utility for item in signals if item.utility is not None]
    if not utilities:
        return None
    return sum(utilities) / float(len(utilities))


def regression_rate(signals: list[HarnessOutcomeSignal]) -> float:
    if not signals:
        return 0.0
    flagged = sum(1 for item in signals if item.regression_flags)
    return flagged / float(len(signals))


def check_utility_trend(
    *,
    candidate_signals: list[HarnessOutcomeSignal],
    baseline_signals: list[HarnessOutcomeSignal],
    context: VerificationContext,
) -> VerificationCheckResult:
    candidate_mean = mean_utility(candidate_signals)
    baseline_mean = mean_utility(baseline_signals)
    if candidate_mean is None or baseline_mean is None:
        return VerificationCheckResult(
            check_id=VerificationCheckId.UTILITY_TREND,
            passed=False,
            detail="Insufficient utility samples for candidate or baseline window",
            metric_value=candidate_mean,
            baseline_value=baseline_mean,
        )
    if len(candidate_signals) < context.min_run_count:
        return VerificationCheckResult(
            check_id=VerificationCheckId.UTILITY_TREND,
            passed=False,
            detail=(
                f"Candidate run count {len(candidate_signals)} "
                f"below minimum {context.min_run_count}"
            ),
            metric_value=candidate_mean,
            baseline_value=baseline_mean,
        )
    delta = candidate_mean - baseline_mean
    passed = delta >= context.min_improvement_delta
    return VerificationCheckResult(
        check_id=VerificationCheckId.UTILITY_TREND,
        passed=passed,
        detail=(
            f"Candidate utility {candidate_mean:.4f} vs baseline {baseline_mean:.4f} "
            f"(delta={delta:.4f}, min_delta={context.min_improvement_delta:.4f})"
        ),
        metric_value=candidate_mean,
        baseline_value=baseline_mean,
    )


def check_eval_registry_trend(
    *,
    evaluation_trend: EvaluationRegistryTrendReport | None,
    context: VerificationContext,
) -> VerificationCheckResult:
    if evaluation_trend is None or not evaluation_trend.comparisons:
        return VerificationCheckResult(
            check_id=VerificationCheckId.EVAL_REGISTRY,
            passed=True,
            detail="No evaluation registry trend supplied; check skipped",
        )
    latest = evaluation_trend.comparisons[-1]
    passed = latest.delta >= context.min_improvement_delta
    return VerificationCheckResult(
        check_id=VerificationCheckId.EVAL_REGISTRY,
        passed=passed,
        detail=(
            f"Eval registry delta {latest.delta:.4f} "
            f"({latest.release_from} -> {latest.release_to})"
        ),
        metric_value=latest.pass_rate_to,
        baseline_value=latest.pass_rate_from,
    )


def check_regression_rate(
    *,
    candidate_signals: list[HarnessOutcomeSignal],
    baseline_signals: list[HarnessOutcomeSignal],
    context: VerificationContext,
) -> VerificationCheckResult:
    candidate_rate = regression_rate(candidate_signals)
    baseline_rate = regression_rate(baseline_signals)
    delta = candidate_rate - baseline_rate
    passed = delta <= context.max_regression_rate_delta
    return VerificationCheckResult(
        check_id=VerificationCheckId.REGRESSION_RATE,
        passed=passed,
        detail=(
            f"Regression rate candidate={candidate_rate:.4f} baseline={baseline_rate:.4f} "
            f"(delta={delta:.4f}, max_delta={context.max_regression_rate_delta:.4f})"
        ),
        metric_value=candidate_rate,
        baseline_value=baseline_rate,
    )


def check_cost_budget(
    *,
    candidate_signals: list[HarnessOutcomeSignal],
    budget_envelopes: list[BudgetEnvelope],
    context: VerificationContext,
) -> VerificationCheckResult:
    if budget_envelopes:
        report = evaluate_budget_envelopes(budget_envelopes)
        over_budget = [item for item in report.decisions if not item.within_budget]
        if over_budget:
            scope = over_budget[0].scope_id
            return VerificationCheckResult(
                check_id=VerificationCheckId.COST_BUDGET,
                passed=False,
                detail=f"Budget envelope exceeded for scope '{scope}'",
            )

    if not candidate_signals:
        return VerificationCheckResult(
            check_id=VerificationCheckId.COST_BUDGET,
            passed=True,
            detail="No candidate signals; cost check skipped",
        )
    peak_cost = max(item.cost_normalized for item in candidate_signals)
    passed = peak_cost <= context.max_cost_normalized
    return VerificationCheckResult(
        check_id=VerificationCheckId.COST_BUDGET,
        passed=passed,
        detail=(
            f"Peak normalized cost {peak_cost:.4f} "
            f"(max={context.max_cost_normalized:.4f})"
        ),
        metric_value=peak_cost,
        baseline_value=context.max_cost_normalized,
    )


def check_security_adversarial(
    checker: SecurityAdversarialBaselineChecker,
) -> VerificationCheckResult:
    passed = checker.evaluate()
    return VerificationCheckResult(
        check_id=VerificationCheckId.SECURITY_ADVERSARIAL,
        passed=passed,
        detail="V-SEC adversarial harness baseline green" if passed else "Adversarial suite failed",
        metric_value=1.0 if passed else 0.0,
        baseline_value=1.0,
    )
