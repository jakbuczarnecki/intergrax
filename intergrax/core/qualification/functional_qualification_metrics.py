# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Generic metrics engine for functional qualification (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.qualification.functional_diagnostic_expectation import QualificationComparisonResult
from intergrax.core.qualification.functional_qualification_case import (
    QualificationNormalizedCaseResult,
    QualificationRepeatabilityGroup,
    functional_failure_detected,
    repeatability_passes,
    stage_matches_case,
)
from intergrax.runtime.diagnostics.functional_diagnostic_analysis import FunctionalDiagnosticCheckStatus


@dataclass(frozen=True, slots=True)
class QualificationPluginMetrics:
    total_cases: int
    matched_cases: int
    mismatched_cases: int
    false_positives: int
    false_negatives: int
    inconclusive_correct_cases: int
    stage_matched_cases: int
    stage_accuracy_percent: float
    inconclusive_accuracy_percent: float
    repeatability_pass: bool
    full_case_match_rate: float
    total_attempts: int = 0
    prerequisite_misses: int = 0
    cases_requiring_retry: int = 0
    prerequisite_exhaustions: int = 0
    prerequisite_success_rate: float = 100.0


def compute_qualification_metrics(
    cases: tuple[QualificationNormalizedCaseResult, ...],
    *,
    repeatability_groups: tuple[QualificationRepeatabilityGroup, ...],
) -> QualificationPluginMetrics:
    total = len(cases)
    matched = sum(1 for item in cases if item.comparison.result is QualificationComparisonResult.MATCH)
    mismatched = total - matched
    stage_matched = sum(
        1
        for item in cases
        if stage_matches_case(
            expected_first_failed_check=item.expected_first_failed_check,
            actual_first_failed_check=item.diag_first_failed_check,
        )
    )
    false_positives = sum(
        1
        for item in cases
        if item.is_healthy_case
        and item.comparison.result is QualificationComparisonResult.MISMATCH
        and any(
            mismatch.actual_status is FunctionalDiagnosticCheckStatus.PROVEN_FAIL
            for mismatch in item.comparison.check_mismatches
        )
    )
    functional_failure_ground_truth = [
        item for item in cases if item.expects_functional_failure and not item.expects_inconclusive
    ]
    false_negatives = sum(
        1
        for item in functional_failure_ground_truth
        if not functional_failure_detected(
            functional_outcome=item.functional_outcome,
            operator_outcome=item.operator_outcome,
        )
    )
    inconclusive_correct = sum(
        1
        for item in cases
        if item.expects_inconclusive and item.comparison.result is QualificationComparisonResult.MATCH
    )
    stage_accuracy = (stage_matched / total * 100.0) if total else 0.0
    inconclusive_accuracy = 100.0 if inconclusive_correct >= 1 else 0.0
    repeatability = repeatability_passes(repeatability_groups)
    full_case_match_rate = (matched / total * 100.0) if total else 0.0
    attempt_totals = sum(item.attempt_count for item in cases)
    prerequisite_misses = sum(item.prerequisite_miss_count for item in cases)
    cases_requiring_retry = sum(1 for item in cases if item.attempt_count > 1)
    prerequisite_exhaustions = sum(1 for item in cases if item.prerequisite_exhausted)
    satisfied_attempts = attempt_totals - prerequisite_misses
    prerequisite_success_rate = (
        (satisfied_attempts / attempt_totals * 100.0) if attempt_totals else 100.0
    )
    return QualificationPluginMetrics(
        total_cases=total,
        matched_cases=matched,
        mismatched_cases=mismatched,
        false_positives=false_positives,
        false_negatives=false_negatives,
        inconclusive_correct_cases=inconclusive_correct,
        stage_matched_cases=stage_matched,
        stage_accuracy_percent=stage_accuracy,
        inconclusive_accuracy_percent=inconclusive_accuracy,
        repeatability_pass=repeatability,
        full_case_match_rate=full_case_match_rate,
        total_attempts=attempt_totals,
        prerequisite_misses=prerequisite_misses,
        cases_requiring_retry=cases_requiring_retry,
        prerequisite_exhaustions=prerequisite_exhaustions,
        prerequisite_success_rate=prerequisite_success_rate,
    )


def metrics_pass(metrics: QualificationPluginMetrics) -> bool:
    return (
        metrics.mismatched_cases == 0
        and metrics.false_positives == 0
        and metrics.false_negatives == 0
        and metrics.repeatability_pass
    )


__all__ = [
    "QualificationPluginMetrics",
    "compute_qualification_metrics",
    "metrics_pass",
]
