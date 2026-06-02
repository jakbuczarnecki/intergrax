# © Artur Czarnecki. All rights reserved.

"""Evaluation registry trend and comparison contracts (Phase V-EVAL.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.evaluation_automation import AutomatedEvaluationReport


class EvaluationReleaseSnapshot(BaseModel):
    release_id: str
    automated_report: AutomatedEvaluationReport


class EvaluationComparisonSummary(BaseModel):
    release_from: str
    release_to: str
    pass_rate_from: float
    pass_rate_to: float
    delta: float


class EvaluationRegistryTrendReport(BaseModel):
    schema_version: str = "1.0.0"
    snapshots: list[EvaluationReleaseSnapshot] = Field(default_factory=list)
    comparisons: list[EvaluationComparisonSummary] = Field(default_factory=list)


def build_evaluation_registry_trend_report(
    snapshots: list[EvaluationReleaseSnapshot],
) -> EvaluationRegistryTrendReport:
    comparisons: list[EvaluationComparisonSummary] = []
    for index in range(1, len(snapshots)):
        previous = snapshots[index - 1]
        current = snapshots[index]
        previous_rate = _pass_rate(previous.automated_report)
        current_rate = _pass_rate(current.automated_report)
        comparisons.append(
            EvaluationComparisonSummary(
                release_from=previous.release_id,
                release_to=current.release_id,
                pass_rate_from=previous_rate,
                pass_rate_to=current_rate,
                delta=current_rate - previous_rate,
            )
        )
    return EvaluationRegistryTrendReport(snapshots=snapshots, comparisons=comparisons)


def _pass_rate(report: AutomatedEvaluationReport) -> float:
    if not report.records:
        return 0.0
    passed = sum(1 for record in report.records if record.final_passed)
    return float(passed) / float(len(report.records))
