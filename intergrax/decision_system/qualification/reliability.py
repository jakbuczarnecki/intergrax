# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Decision qualification reliability aggregation (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.axis_outcome import DecisionQualificationAxisOutcome
from intergrax.decision_system.qualification.run_result import DecisionQualificationRunResult
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory


class DecisionReliabilitySummaryConsistencyError(ValueError):
    """Fail-closed when aggregated reliability counts are inconsistent."""


@dataclass(frozen=True, slots=True)
class DecisionReliabilitySummary:
    total_runs: int
    platform_evaluable_count: int
    platform_pass_count: int
    platform_failure_count: int
    model_evaluable_count: int
    model_pass_count: int
    model_failure_count: int
    evaluator_evaluable_count: int
    evaluator_pass_count: int
    evaluator_fail_count: int
    provider_infra_failure_count: int
    environment_failure_count: int
    observability_gap_count: int

    def __post_init__(self) -> None:
        if self.total_runs < 0:
            raise DecisionReliabilitySummaryConsistencyError("total_runs must be non-negative")
        _validate_axis_counts(
            axis_name="platform",
            total_runs=self.total_runs,
            evaluable_count=self.platform_evaluable_count,
            pass_count=self.platform_pass_count,
            failure_count=self.platform_failure_count,
        )
        _validate_axis_counts(
            axis_name="model",
            total_runs=self.total_runs,
            evaluable_count=self.model_evaluable_count,
            pass_count=self.model_pass_count,
            failure_count=self.model_failure_count,
        )
        _validate_axis_counts(
            axis_name="evaluator",
            total_runs=self.total_runs,
            evaluable_count=self.evaluator_evaluable_count,
            pass_count=self.evaluator_pass_count,
            failure_count=self.evaluator_fail_count,
        )

    @property
    def platform_not_evaluable_count(self) -> int:
        return self.total_runs - self.platform_evaluable_count

    @property
    def model_not_evaluable_count(self) -> int:
        return self.total_runs - self.model_evaluable_count

    @property
    def evaluator_not_evaluable_count(self) -> int:
        return self.total_runs - self.evaluator_evaluable_count

    @property
    def platform_reliability(self) -> float | None:
        if self.platform_evaluable_count == 0:
            return None
        return self.platform_pass_count / self.platform_evaluable_count

    @property
    def model_reliability(self) -> float | None:
        if self.model_evaluable_count == 0:
            return None
        return self.model_pass_count / self.model_evaluable_count

    @property
    def evaluator_pass_rate(self) -> float | None:
        if self.evaluator_evaluable_count == 0:
            return None
        return self.evaluator_pass_count / self.evaluator_evaluable_count

    @property
    def platform_evaluation_coverage(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.platform_evaluable_count / self.total_runs

    @property
    def model_evaluation_coverage(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.model_evaluable_count / self.total_runs

    @property
    def evaluator_evaluation_coverage(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.evaluator_evaluable_count / self.total_runs


def _validate_axis_counts(
    *,
    axis_name: str,
    total_runs: int,
    evaluable_count: int,
    pass_count: int,
    failure_count: int,
) -> None:
    if evaluable_count > total_runs:
        raise DecisionReliabilitySummaryConsistencyError(
            f"{axis_name} evaluable_count exceeds total_runs"
        )
    if pass_count < 0 or failure_count < 0:
        raise DecisionReliabilitySummaryConsistencyError(
            f"{axis_name} pass/failure counts must be non-negative"
        )
    if pass_count + failure_count != evaluable_count:
        raise DecisionReliabilitySummaryConsistencyError(
            f"{axis_name} pass_count + failure_count must equal evaluable_count"
        )


def _axis_pass(outcome: DecisionQualificationAxisOutcome) -> bool:
    return outcome is DecisionQualificationAxisOutcome.PASS


def _axis_fail(outcome: DecisionQualificationAxisOutcome) -> bool:
    return outcome is DecisionQualificationAxisOutcome.FAIL


def _axis_evaluable(outcome: DecisionQualificationAxisOutcome) -> bool:
    return outcome is not DecisionQualificationAxisOutcome.NOT_EVALUABLE


def aggregate_decision_reliability(
    run_results: tuple[DecisionQualificationRunResult, ...],
) -> DecisionReliabilitySummary:
    total_runs = len(run_results)
    platform_pass_count = sum(1 for item in run_results if _axis_pass(item.platform_outcome))
    platform_failure_count = sum(1 for item in run_results if _axis_fail(item.platform_outcome))
    platform_evaluable_count = sum(
        1 for item in run_results if _axis_evaluable(item.platform_outcome)
    )
    model_pass_count = sum(1 for item in run_results if _axis_pass(item.model_outcome))
    model_failure_count = sum(1 for item in run_results if _axis_fail(item.model_outcome))
    model_evaluable_count = sum(1 for item in run_results if _axis_evaluable(item.model_outcome))
    evaluator_pass_count = sum(1 for item in run_results if _axis_pass(item.evaluator_outcome))
    evaluator_fail_count = sum(1 for item in run_results if _axis_fail(item.evaluator_outcome))
    evaluator_evaluable_count = sum(
        1 for item in run_results if _axis_evaluable(item.evaluator_outcome)
    )

    provider_infra_failure_count = 0
    environment_failure_count = 0
    observability_gap_count = 0
    for item in run_results:
        if item.classification is None:
            continue
        category = item.classification.category
        if category is DecisionFailureCategory.PROVIDER_INFRASTRUCTURE:
            provider_infra_failure_count += 1
        elif category is DecisionFailureCategory.ENVIRONMENT:
            environment_failure_count += 1
        elif category is DecisionFailureCategory.OBSERVABILITY_GAP:
            observability_gap_count += 1

    return DecisionReliabilitySummary(
        total_runs=total_runs,
        platform_evaluable_count=platform_evaluable_count,
        platform_pass_count=platform_pass_count,
        platform_failure_count=platform_failure_count,
        model_evaluable_count=model_evaluable_count,
        model_pass_count=model_pass_count,
        model_failure_count=model_failure_count,
        evaluator_evaluable_count=evaluator_evaluable_count,
        evaluator_pass_count=evaluator_pass_count,
        evaluator_fail_count=evaluator_fail_count,
        provider_infra_failure_count=provider_infra_failure_count,
        environment_failure_count=environment_failure_count,
        observability_gap_count=observability_gap_count,
    )
