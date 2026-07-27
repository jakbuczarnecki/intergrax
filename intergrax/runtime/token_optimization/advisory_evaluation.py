# © Artur Czarnecki. All rights reserved.

"""Advisory recommendation evaluation and redaction-safe reporting (TOKEN-7B)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from intergrax.runtime.token_optimization.advisory import recommend_token_optimization_action
from intergrax.runtime.token_optimization.contracts import (
    TokenOptimizationAdvisorySignal,
    TokenOptimizationRecommendationAction,
    TokenOptimizationRecommendationConfidence,
    TokenOptimizationRecommendationReason,
    TokenOptimizationSourceType,
    TokenOptimizationStrategyKind,
)

_FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
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


def _validate_non_empty_stripped(value: str, field_name: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{field_name} must be non-empty after stripping")
    return stripped


def _validate_string_metadata(metadata: Mapping[str, str]) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise ValueError("metadata keys and values must be strings")
        if key in _FORBIDDEN_METADATA_KEYS:
            raise ValueError(f"metadata must not contain raw-content-like key: {key}")
        validated[key] = value
    return validated


def _enum_value(value: StrEnum | None) -> str | None:
    if value is None:
        return None
    return value.value


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryEvaluationCase:
    """One advisory evaluation case with expected recommendation outcomes."""

    case_id: str
    title: str
    signal: TokenOptimizationAdvisorySignal
    expected_action: TokenOptimizationRecommendationAction
    expected_reason: TokenOptimizationRecommendationReason
    expected_confidence: TokenOptimizationRecommendationConfidence | None = None
    expected_auto_apply_allowed: bool = False
    expected_raw_content_included: bool = False
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _validate_non_empty_stripped(self.case_id, "case_id"))
        object.__setattr__(self, "title", _validate_non_empty_stripped(self.title, "title"))
        if self.expected_auto_apply_allowed:
            raise ValueError("expected_auto_apply_allowed must remain False")
        if self.expected_raw_content_included:
            raise ValueError("expected_raw_content_included must remain False")
        object.__setattr__(
            self,
            "metadata",
            _validate_string_metadata(self.metadata),
        )


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryEvaluationResult:
    """Redaction-safe per-case advisory evaluation outcome."""

    case_id: str
    passed: bool
    actual_action: TokenOptimizationRecommendationAction
    expected_action: TokenOptimizationRecommendationAction
    actual_reason: TokenOptimizationRecommendationReason
    expected_reason: TokenOptimizationRecommendationReason
    actual_confidence: TokenOptimizationRecommendationConfidence
    expected_confidence: TokenOptimizationRecommendationConfidence | None
    auto_apply_allowed: bool
    raw_content_included: bool
    recommendation_source_type: TokenOptimizationSourceType
    recommendation_strategy_kind: TokenOptimizationStrategyKind | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _validate_non_empty_stripped(self.case_id, "case_id"))
        if self.auto_apply_allowed:
            raise ValueError("auto_apply_allowed must remain False")
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryEvaluationSummary:
    """Aggregate advisory evaluation counts (redaction-safe scalars only)."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    manual_review_recommendations: int
    insufficient_data_recommendations: int
    non_auto_apply_recommendations: int
    raw_content_safe_results: int

    def __post_init__(self) -> None:
        counts = (
            self.total_cases,
            self.passed_cases,
            self.failed_cases,
            self.manual_review_recommendations,
            self.insufficient_data_recommendations,
            self.non_auto_apply_recommendations,
            self.raw_content_safe_results,
        )
        if any(count < 0 for count in counts):
            raise ValueError("all counts must be >= 0")
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("passed_cases + failed_cases must equal total_cases")
        if self.non_auto_apply_recommendations > self.total_cases:
            raise ValueError("non_auto_apply_recommendations must be <= total_cases")
        if self.raw_content_safe_results > self.total_cases:
            raise ValueError("raw_content_safe_results must be <= total_cases")


@dataclass(frozen=True, slots=True)
class TokenOptimizationAdvisoryEvaluationReport:
    """Redaction-safe advisory evaluation report artifact."""

    summary: TokenOptimizationAdvisoryEvaluationSummary
    results: tuple[TokenOptimizationAdvisoryEvaluationResult, ...]
    report_kind: str = "token_optimization_advisory_evaluation"
    raw_content_included: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "report_kind",
            _validate_non_empty_stripped(self.report_kind, "report_kind"),
        )
        if self.raw_content_included:
            raise ValueError("raw_content_included must remain False")
        if len(self.results) != self.summary.total_cases:
            raise ValueError("len(results) must equal summary.total_cases")


def evaluate_advisory_recommendation_case(
    case: TokenOptimizationAdvisoryEvaluationCase,
) -> TokenOptimizationAdvisoryEvaluationResult:
    """Evaluate one advisory case against the policy-only recommender."""
    recommendation = recommend_token_optimization_action(case.signal)
    if recommendation.auto_apply_allowed:
        raise ValueError("auto_apply_allowed must remain False")
    if recommendation.raw_content_included:
        raise ValueError("raw_content_included must remain False")

    action_matches = recommendation.action is case.expected_action
    reason_matches = recommendation.reason is case.expected_reason
    confidence_matches = (
        case.expected_confidence is None
        or recommendation.confidence is case.expected_confidence
    )
    passed = action_matches and reason_matches and confidence_matches

    return TokenOptimizationAdvisoryEvaluationResult(
        case_id=case.case_id,
        passed=passed,
        actual_action=recommendation.action,
        expected_action=case.expected_action,
        actual_reason=recommendation.reason,
        expected_reason=case.expected_reason,
        actual_confidence=recommendation.confidence,
        expected_confidence=case.expected_confidence,
        auto_apply_allowed=recommendation.auto_apply_allowed,
        raw_content_included=recommendation.raw_content_included,
        recommendation_source_type=recommendation.source_type,
        recommendation_strategy_kind=recommendation.strategy_kind,
    )


def evaluate_advisory_recommendation_cases(
    cases: Sequence[TokenOptimizationAdvisoryEvaluationCase],
) -> TokenOptimizationAdvisoryEvaluationReport:
    """Evaluate advisory cases in input order and build a redaction-safe report."""
    seen_case_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case.case_id}")
        seen_case_ids.add(case.case_id)

    results = tuple(evaluate_advisory_recommendation_case(case) for case in cases)
    passed_cases = sum(1 for result in results if result.passed)
    failed_cases = len(results) - passed_cases
    summary = TokenOptimizationAdvisoryEvaluationSummary(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        manual_review_recommendations=sum(
            1
            for result in results
            if result.actual_action is TokenOptimizationRecommendationAction.REQUIRE_MANUAL_REVIEW
        ),
        insufficient_data_recommendations=sum(
            1
            for result in results
            if result.actual_action is TokenOptimizationRecommendationAction.INSUFFICIENT_DATA
        ),
        non_auto_apply_recommendations=sum(
            1 for result in results if not result.auto_apply_allowed
        ),
        raw_content_safe_results=sum(
            1 for result in results if not result.raw_content_included
        ),
    )
    return TokenOptimizationAdvisoryEvaluationReport(summary=summary, results=results)


def _result_to_dict(result: TokenOptimizationAdvisoryEvaluationResult) -> dict[str, object]:
    return {
        "case_id": result.case_id,
        "passed": result.passed,
        "actual_action": _enum_value(result.actual_action),
        "expected_action": _enum_value(result.expected_action),
        "actual_reason": _enum_value(result.actual_reason),
        "expected_reason": _enum_value(result.expected_reason),
        "actual_confidence": _enum_value(result.actual_confidence),
        "expected_confidence": _enum_value(result.expected_confidence),
        "auto_apply_allowed": result.auto_apply_allowed,
        "raw_content_included": result.raw_content_included,
        "recommendation_source_type": _enum_value(result.recommendation_source_type),
        "recommendation_strategy_kind": _enum_value(result.recommendation_strategy_kind),
    }


def _summary_to_dict(summary: TokenOptimizationAdvisoryEvaluationSummary) -> dict[str, object]:
    return {
        "total_cases": summary.total_cases,
        "passed_cases": summary.passed_cases,
        "failed_cases": summary.failed_cases,
        "manual_review_recommendations": summary.manual_review_recommendations,
        "insufficient_data_recommendations": summary.insufficient_data_recommendations,
        "non_auto_apply_recommendations": summary.non_auto_apply_recommendations,
        "raw_content_safe_results": summary.raw_content_safe_results,
    }


def token_optimization_advisory_report_to_dict(
    report: TokenOptimizationAdvisoryEvaluationReport,
) -> dict[str, object]:
    """Serialize an advisory evaluation report for JSON output (redaction-safe)."""
    return {
        "report_kind": report.report_kind,
        "raw_content_included": report.raw_content_included,
        "summary": _summary_to_dict(report.summary),
        "results": [_result_to_dict(result) for result in report.results],
    }


def format_token_optimization_advisory_report(
    report: TokenOptimizationAdvisoryEvaluationReport,
) -> str:
    """Human-readable advisory evaluation report (deterministic, redaction-safe)."""
    summary = report.summary
    lines = [
        f"report_kind={report.report_kind}",
        (
            f"total_cases={summary.total_cases} "
            f"passed_cases={summary.passed_cases} "
            f"failed_cases={summary.failed_cases} "
            f"manual_review_recommendations={summary.manual_review_recommendations} "
            f"insufficient_data_recommendations={summary.insufficient_data_recommendations} "
            f"non_auto_apply_recommendations={summary.non_auto_apply_recommendations} "
            f"raw_content_safe_results={summary.raw_content_safe_results}"
        ),
    ]
    for result in report.results:
        status = "PASS" if result.passed else "FAIL"
        lines.append(
            f"  [{status}] case_id={result.case_id} "
            f"actual_action={_enum_value(result.actual_action)} "
            f"expected_action={_enum_value(result.expected_action)} "
            f"actual_reason={_enum_value(result.actual_reason)} "
            f"expected_reason={_enum_value(result.expected_reason)}"
        )
    return "\n".join(lines)
