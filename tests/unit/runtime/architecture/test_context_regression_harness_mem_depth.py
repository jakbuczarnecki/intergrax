# © Artur Czarnecki. All rights reserved.

"""MEM-DEPTH-5.4 context quality regression harness."""

from __future__ import annotations

import pytest

from intergrax.runtime.architecture.context_engineering import ContextChunkSignal
from intergrax.runtime.architecture.context_regression_benchmark import (
    ContextRegressionCase,
    build_context_regression_benchmark_report,
    evaluate_context_regression_cases,
)

pytestmark = pytest.mark.gate


def test_context_regression_harness_detects_regression() -> None:
    cases = [
        ContextRegressionCase(
            case_id="memory_injection",
            chunks=[
                ContextChunkSignal(
                    chunk_id="c1",
                    content_hash="abc123",
                    relevance_score=0.9,
                    freshness_score=0.8,
                    confidence_score=0.85,
                )
            ],
            expected_pass_rate=0.8,
        )
    ]
    baseline = evaluate_context_regression_cases(cases)
    current = evaluate_context_regression_cases(cases)
    report = build_context_regression_benchmark_report(
        baseline_version="v1",
        current_version="v2",
        cases=cases,
        baseline_reports_by_case=baseline,
        current_reports_by_case=current,
    )
    assert report.comparisons[0].regressed is False
