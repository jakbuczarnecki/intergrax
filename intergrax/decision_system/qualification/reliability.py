# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Decision qualification reliability aggregation (DS-E2E-14.3)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.decision_system.qualification.run_result import DecisionQualificationRunResult
from intergrax.decision_system.qualification.taxonomy import DecisionFailureCategory


@dataclass(frozen=True, slots=True)
class DecisionReliabilitySummary:
    total_runs: int
    platform_pass_count: int
    platform_failure_count: int
    model_pass_count: int
    model_failure_count: int
    evaluator_pass_count: int
    provider_infra_failure_count: int
    environment_failure_count: int
    observability_gap_count: int

    @property
    def platform_reliability(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.platform_pass_count / self.total_runs

    @property
    def model_reliability(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.model_pass_count / self.total_runs

    @property
    def evaluator_pass_rate(self) -> float | None:
        if self.total_runs == 0:
            return None
        return self.evaluator_pass_count / self.total_runs


def aggregate_decision_reliability(
    run_results: tuple[DecisionQualificationRunResult, ...],
) -> DecisionReliabilitySummary:
    total_runs = len(run_results)
    platform_pass_count = sum(1 for item in run_results if item.platform_contract_passed)
    model_pass_count = sum(1 for item in run_results if item.model_behavior_passed)
    evaluator_pass_count = sum(1 for item in run_results if item.evaluator_passed)
    platform_failure_count = total_runs - platform_pass_count
    model_failure_count = total_runs - model_pass_count

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
        platform_pass_count=platform_pass_count,
        platform_failure_count=platform_failure_count,
        model_pass_count=model_pass_count,
        model_failure_count=model_failure_count,
        evaluator_pass_count=evaluator_pass_count,
        provider_infra_failure_count=provider_infra_failure_count,
        environment_failure_count=environment_failure_count,
        observability_gap_count=observability_gap_count,
    )
