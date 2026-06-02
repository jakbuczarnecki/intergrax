# © Artur Czarnecki. All rights reserved.

"""Context regression benchmark contracts (Phase V-CE.3)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.context_engineering import (
    ContextEngineeringReport,
    evaluate_context_engineering,
    ContextChunkSignal,
)


class ContextRegressionCase(BaseModel):
    case_id: str
    chunks: list[ContextChunkSignal] = Field(default_factory=list)
    expected_pass_rate: float


class ContextRegressionComparison(BaseModel):
    case_id: str
    baseline_pass_rate: float
    current_pass_rate: float
    delta: float
    regressed: bool


class ContextRegressionBenchmarkReport(BaseModel):
    schema_version: str = "1.0.0"
    baseline_version: str
    current_version: str
    comparisons: list[ContextRegressionComparison] = Field(default_factory=list)


def build_context_regression_benchmark_report(
    *,
    baseline_version: str,
    current_version: str,
    cases: list[ContextRegressionCase],
    baseline_reports_by_case: dict[str, ContextEngineeringReport],
    current_reports_by_case: dict[str, ContextEngineeringReport],
    regression_tolerance: float = 0.05,
) -> ContextRegressionBenchmarkReport:
    comparisons: list[ContextRegressionComparison] = []
    for case in cases:
        baseline_report = baseline_reports_by_case[case.case_id]
        current_report = current_reports_by_case[case.case_id]
        baseline_rate = _pass_rate(baseline_report)
        current_rate = _pass_rate(current_report)
        delta = current_rate - baseline_rate
        comparisons.append(
            ContextRegressionComparison(
                case_id=case.case_id,
                baseline_pass_rate=baseline_rate,
                current_pass_rate=current_rate,
                delta=delta,
                regressed=delta < -regression_tolerance,
            )
        )
    return ContextRegressionBenchmarkReport(
        baseline_version=baseline_version,
        current_version=current_version,
        comparisons=comparisons,
    )


def evaluate_context_regression_cases(
    cases: list[ContextRegressionCase],
) -> dict[str, ContextEngineeringReport]:
    return {
        case.case_id: evaluate_context_engineering(chunks=case.chunks)
        for case in cases
    }


def _pass_rate(report: ContextEngineeringReport) -> float:
    if not report.records:
        return 0.0
    passed = sum(1 for record in report.records if record.passed)
    return float(passed) / float(len(report.records))
