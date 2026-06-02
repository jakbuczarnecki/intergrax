from __future__ import annotations

from intergrax.runtime.architecture.context_engineering import ContextChunkSignal
from intergrax.runtime.architecture.context_regression_benchmark import (
    ContextRegressionCase,
    build_context_regression_benchmark_report,
    evaluate_context_regression_cases,
)


def test_context_regression_benchmark_detects_regression() -> None:
    cases = [
        ContextRegressionCase(
            case_id="case-1",
            chunks=[
                ContextChunkSignal(
                    chunk_id="c1",
                    content_hash="h1",
                    relevance_score=0.9,
                    freshness_score=0.9,
                    confidence_score=0.9,
                )
            ],
            expected_pass_rate=1.0,
        )
    ]
    baseline = evaluate_context_regression_cases(cases)
    degraded_cases = [
        ContextRegressionCase(
            case_id="case-1",
            chunks=[
                ContextChunkSignal(
                    chunk_id="c1",
                    content_hash="h1",
                    relevance_score=0.3,
                    freshness_score=0.3,
                    confidence_score=0.3,
                )
            ],
            expected_pass_rate=0.0,
        )
    ]
    current = evaluate_context_regression_cases(degraded_cases)
    report = build_context_regression_benchmark_report(
        baseline_version="v1",
        current_version="v2",
        cases=cases,
        baseline_reports_by_case=baseline,
        current_reports_by_case=current,
    )
    assert report.comparisons[0].regressed is True
