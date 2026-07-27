# © Artur Czarnecki. All rights reserved.

"""TOKEN-7B: advisory evaluation and redaction-safe reporting tests."""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any

import pytest

from intergrax.runtime.token_optimization.advisory_evaluation import (
    TokenOptimizationAdvisoryEvaluationCase,
    TokenOptimizationAdvisoryEvaluationReport,
    TokenOptimizationAdvisoryEvaluationResult,
    TokenOptimizationAdvisoryEvaluationSummary,
    evaluate_advisory_recommendation_case,
    evaluate_advisory_recommendation_cases,
    format_token_optimization_advisory_report,
    token_optimization_advisory_report_to_dict,
)
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)
from tests.fixtures.token_optimization.advisory_evaluation_corpus import (
    ADVISORY_EVALUATION_CORPUS,
    ADVISORY_EVALUATION_SYNTHETIC_CORPUS_MARKER,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_FORBIDDEN_SAVINGS_FIELD_NAMES = frozenset(
    {
        "saved_tokens",
        "optimized_tokens",
        "baseline_tokens",
        "compressed_tokens",
    }
)

_FORBIDDEN_CONTENT_FIELD_NAMES = frozenset(
    {
        "prompt",
        "content",
        "context",
        "evidence",
        "tool_output",
        "raw_payload",
        "document_text",
        "source_text",
    }
)

_REQUIRED_CORPUS_CASE_IDS = frozenset(
    {
        "advisory_eval.policy_review_requires_manual_review",
        "advisory_eval.protected_region_risk_escalates_to_full_context",
        "advisory_eval.quality_regression_escalates_to_full_context",
        "advisory_eval.regression_gate_failed_requires_review",
        "advisory_eval.insufficient_data",
        "advisory_eval.high_fallback_disables_strategy",
        "advisory_eval.hot_stable_cache_preserves_prefix",
        "advisory_eval.invalidated_cache_requires_review",
        "advisory_eval.dynamic_tail_reduction_preferred",
        "advisory_eval.measured_safe_savings_enables_strategy",
        "advisory_eval.low_savings_keeps_current",
        "advisory_eval.regression_gate_passed_keeps_current",
    }
)


def _sample_case() -> TokenOptimizationAdvisoryEvaluationCase:
    return ADVISORY_EVALUATION_CORPUS[0]


def _collect_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.add(str(key))
            keys.update(_collect_keys(nested))
    elif isinstance(value, (list, tuple)):
        for item in value:
            keys.update(_collect_keys(item))
    return keys


def test_evaluation_case_validates_required_fields() -> None:
    with pytest.raises(ValueError, match="case_id must be non-empty"):
        TokenOptimizationAdvisoryEvaluationCase(
            case_id="   ",
            title="title",
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
            ),
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
        )
    with pytest.raises(ValueError, match="title must be non-empty"):
        TokenOptimizationAdvisoryEvaluationCase(
            case_id="case-1",
            title="   ",
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
            ),
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
        )


def test_evaluation_case_rejects_raw_content_metadata_keys() -> None:
    for key in _FORBIDDEN_CONTENT_FIELD_NAMES:
        with pytest.raises(ValueError, match="metadata must not contain raw-content-like key"):
            TokenOptimizationAdvisoryEvaluationCase(
                case_id="case-1",
                title="title",
                signal=TokenOptimizationAdvisorySignal(
                    source_type=TokenOptimizationSourceType.PROMPT,
                ),
                expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
                expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
                metadata={key: "value"},
            )


def test_evaluation_case_rejects_auto_apply_expectation() -> None:
    with pytest.raises(ValueError, match="expected_auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryEvaluationCase(
            case_id="case-1",
            title="title",
            signal=TokenOptimizationAdvisorySignal(
                source_type=TokenOptimizationSourceType.PROMPT,
            ),
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            expected_auto_apply_allowed=True,
        )


def test_evaluation_result_rejects_auto_apply() -> None:
    with pytest.raises(ValueError, match="auto_apply_allowed must remain False"):
        TokenOptimizationAdvisoryEvaluationResult(
            case_id="case-1",
            passed=True,
            actual_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            actual_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            actual_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            auto_apply_allowed=True,
            raw_content_included=False,
            recommendation_source_type=TokenOptimizationSourceType.PROMPT,
        )


def test_evaluation_result_rejects_raw_content_flag() -> None:
    with pytest.raises(ValueError, match="raw_content_included must remain False"):
        TokenOptimizationAdvisoryEvaluationResult(
            case_id="case-1",
            passed=True,
            actual_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            expected_action=TokenOptimizationRecommendationAction.KEEP_CURRENT,
            actual_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            expected_reason=TokenOptimizationRecommendationReason.REGRESSION_GATE_PASSED,
            actual_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            expected_confidence=TokenOptimizationRecommendationConfidence.MEDIUM,
            auto_apply_allowed=False,
            raw_content_included=True,
            recommendation_source_type=TokenOptimizationSourceType.PROMPT,
        )


def test_evaluation_summary_validates_counts() -> None:
    with pytest.raises(ValueError, match="all counts must be >= 0"):
        TokenOptimizationAdvisoryEvaluationSummary(
            total_cases=1,
            passed_cases=-1,
            failed_cases=2,
            manual_review_recommendations=0,
            insufficient_data_recommendations=0,
            non_auto_apply_recommendations=0,
            raw_content_safe_results=0,
        )
    with pytest.raises(ValueError, match="passed_cases \\+ failed_cases must equal total_cases"):
        TokenOptimizationAdvisoryEvaluationSummary(
            total_cases=3,
            passed_cases=1,
            failed_cases=1,
            manual_review_recommendations=0,
            insufficient_data_recommendations=0,
            non_auto_apply_recommendations=0,
            raw_content_safe_results=0,
        )
    with pytest.raises(ValueError, match="non_auto_apply_recommendations must be <= total_cases"):
        TokenOptimizationAdvisoryEvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            manual_review_recommendations=0,
            insufficient_data_recommendations=0,
            non_auto_apply_recommendations=2,
            raw_content_safe_results=0,
        )
    with pytest.raises(ValueError, match="raw_content_safe_results must be <= total_cases"):
        TokenOptimizationAdvisoryEvaluationSummary(
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            manual_review_recommendations=0,
            insufficient_data_recommendations=0,
            non_auto_apply_recommendations=1,
            raw_content_safe_results=2,
        )


def test_evaluation_report_validates_result_count() -> None:
    summary = TokenOptimizationAdvisoryEvaluationSummary(
        total_cases=1,
        passed_cases=1,
        failed_cases=0,
        manual_review_recommendations=0,
        insufficient_data_recommendations=0,
        non_auto_apply_recommendations=1,
        raw_content_safe_results=1,
    )
    with pytest.raises(ValueError, match="len\\(results\\) must equal summary.total_cases"):
        TokenOptimizationAdvisoryEvaluationReport(summary=summary, results=())


def test_evaluate_single_case_passes_expected_recommendation() -> None:
    case = _sample_case()
    result = evaluate_advisory_recommendation_case(case)
    assert result.passed is True
    assert result.actual_action is case.expected_action
    assert result.actual_reason is case.expected_reason
    assert result.actual_confidence is case.expected_confidence


def test_evaluate_single_case_fails_unexpected_action() -> None:
    case = TokenOptimizationAdvisoryEvaluationCase(
        case_id="advisory_eval.mismatch",
        title="Mismatch case",
        signal=TokenOptimizationAdvisorySignal(
            source_type=TokenOptimizationSourceType.PROMPT,
            sample_count=0,
        ),
        expected_action=TokenOptimizationRecommendationAction.ENABLE_STRATEGY,
        expected_reason=TokenOptimizationRecommendationReason.MEASURED_SAFE_SAVINGS,
        expected_confidence=TokenOptimizationRecommendationConfidence.HIGH,
    )
    result = evaluate_advisory_recommendation_case(case)
    assert result.passed is False
    assert result.actual_action is TokenOptimizationRecommendationAction.INSUFFICIENT_DATA


def test_evaluate_cases_requires_unique_case_ids() -> None:
    case = _sample_case()
    with pytest.raises(ValueError, match="duplicate case_id"):
        evaluate_advisory_recommendation_cases((case, case))


def test_evaluate_cases_preserves_input_order() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    assert [result.case_id for result in report.results] == [
        case.case_id for case in ADVISORY_EVALUATION_CORPUS
    ]


def test_evaluate_cases_builds_summary_counts() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    summary = report.summary
    assert summary.total_cases == len(ADVISORY_EVALUATION_CORPUS)
    assert summary.passed_cases == len(ADVISORY_EVALUATION_CORPUS)
    assert summary.failed_cases == 0
    assert summary.manual_review_recommendations == 3
    assert summary.insufficient_data_recommendations == 1
    assert summary.non_auto_apply_recommendations == len(ADVISORY_EVALUATION_CORPUS)
    assert summary.raw_content_safe_results == len(ADVISORY_EVALUATION_CORPUS)


def test_evaluation_report_to_dict_is_redaction_safe() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    payload = token_optimization_advisory_report_to_dict(report)
    keys = _collect_keys(payload)
    assert not _FORBIDDEN_CONTENT_FIELD_NAMES.intersection(keys)
    assert not _FORBIDDEN_SAVINGS_FIELD_NAMES.intersection(keys)
    assert payload["raw_content_included"] is False
    assert payload["report_kind"] == "token_optimization_advisory_evaluation"
    assert "signal" not in keys
    assert "recommendation" not in keys
    json.dumps(payload)


def test_format_advisory_report_is_deterministic() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    first = format_token_optimization_advisory_report(report)
    second = format_token_optimization_advisory_report(report)
    assert first == second
    assert "report_kind=token_optimization_advisory_evaluation" in first
    assert "total_cases=12" in first


def test_format_advisory_report_is_redaction_safe() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    text = format_token_optimization_advisory_report(report)
    lowered = text.lower()
    for forbidden in _FORBIDDEN_CONTENT_FIELD_NAMES:
        assert f"{forbidden}=" not in lowered
    for forbidden in _FORBIDDEN_SAVINGS_FIELD_NAMES:
        assert f"{forbidden}=" not in lowered


def test_synthetic_evaluation_corpus_case_ids_are_unique() -> None:
    case_ids = [case.case_id for case in ADVISORY_EVALUATION_CORPUS]
    assert len(case_ids) == len(set(case_ids))


def test_synthetic_evaluation_corpus_has_required_cases() -> None:
    case_ids = {case.case_id for case in ADVISORY_EVALUATION_CORPUS}
    assert _REQUIRED_CORPUS_CASE_IDS.issubset(case_ids)


def test_synthetic_evaluation_corpus_has_marker() -> None:
    assert (
        ADVISORY_EVALUATION_SYNTHETIC_CORPUS_MARKER
        == "SYNTHETIC_ADVISORY_EVALUATION_CORPUS_V1"
    )
    assert all(
        case.metadata.get("synthetic_marker") == ADVISORY_EVALUATION_SYNTHETIC_CORPUS_MARKER
        for case in ADVISORY_EVALUATION_CORPUS
    )


def test_synthetic_evaluation_corpus_all_cases_pass() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    assert report.summary.failed_cases == 0
    assert all(result.passed for result in report.results)


def test_evaluation_results_never_allow_auto_apply() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    assert all(not result.auto_apply_allowed for result in report.results)


def test_evaluation_results_never_include_raw_content() -> None:
    report = evaluate_advisory_recommendation_cases(ADVISORY_EVALUATION_CORPUS)
    result_fields = {
        field.name for field in dataclasses.fields(TokenOptimizationAdvisoryEvaluationResult)
    }
    assert not _FORBIDDEN_CONTENT_FIELD_NAMES.intersection(result_fields)
    assert all(not result.raw_content_included for result in report.results)
