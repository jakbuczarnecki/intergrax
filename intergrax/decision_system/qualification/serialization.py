# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Stable JSON projections for Decision qualification artifacts (DS-E2E-14.3)."""

from __future__ import annotations

from intergrax.decision_system.qualification.classification import DecisionFailureClassification
from intergrax.decision_system.qualification.reliability import DecisionReliabilitySummary
from intergrax.decision_system.qualification.run_result import DecisionQualificationRunResult


def classification_to_dict(
    classification: DecisionFailureClassification,
) -> dict[str, str | bool]:
    return {
        "category": classification.category.value,
        "reason": classification.reason.value,
        "boundary": classification.boundary.value,
        "owner": classification.owner.value,
        "retryability": classification.retryability.value,
        "diagnostic_code": classification.diagnostic_code.value,
        "is_platform_failure": classification.is_platform_failure,
        "is_model_failure": classification.is_model_failure,
        "is_retriable": classification.is_retriable,
    }


def run_result_to_dict(result: DecisionQualificationRunResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_id": str(result.run_id),
        "platform_outcome": result.platform_outcome.value,
        "model_outcome": result.model_outcome.value,
        "evaluator_outcome": result.evaluator_outcome.value,
        "platform_contract_passed": result.platform_contract_passed,
        "model_behavior_passed": result.model_behavior_passed,
        "evaluator_passed": result.evaluator_passed,
        "classification": None,
    }
    if result.classification is not None:
        payload["classification"] = classification_to_dict(result.classification)
    return payload


def reliability_summary_to_dict(summary: DecisionReliabilitySummary) -> dict[str, object]:
    return {
        "total_runs": summary.total_runs,
        "platform_evaluable_count": summary.platform_evaluable_count,
        "platform_pass_count": summary.platform_pass_count,
        "platform_failure_count": summary.platform_failure_count,
        "platform_not_evaluable_count": summary.platform_not_evaluable_count,
        "model_evaluable_count": summary.model_evaluable_count,
        "model_pass_count": summary.model_pass_count,
        "model_failure_count": summary.model_failure_count,
        "model_not_evaluable_count": summary.model_not_evaluable_count,
        "evaluator_evaluable_count": summary.evaluator_evaluable_count,
        "evaluator_pass_count": summary.evaluator_pass_count,
        "evaluator_fail_count": summary.evaluator_fail_count,
        "evaluator_not_evaluable_count": summary.evaluator_not_evaluable_count,
        "provider_infra_failure_count": summary.provider_infra_failure_count,
        "environment_failure_count": summary.environment_failure_count,
        "observability_gap_count": summary.observability_gap_count,
        "platform_reliability": summary.platform_reliability,
        "model_reliability": summary.model_reliability,
        "evaluator_pass_rate": summary.evaluator_pass_rate,
        "platform_evaluation_coverage": summary.platform_evaluation_coverage,
        "model_evaluation_coverage": summary.model_evaluation_coverage,
        "evaluator_evaluation_coverage": summary.evaluator_evaluation_coverage,
    }
