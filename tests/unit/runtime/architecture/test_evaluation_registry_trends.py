from __future__ import annotations

from intergrax.runtime.architecture.evaluation_automation import (
    AutomatedEvaluationRecord,
    AutomatedEvaluationReport,
    AutomatedEvaluatorResult,
    EvaluatorType,
)
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationReleaseSnapshot,
    build_evaluation_registry_trend_report,
)


def _automated_report(*, passed_records: int, failed_records: int) -> AutomatedEvaluationReport:
    records: list[AutomatedEvaluationRecord] = []
    for index in range(passed_records):
        records.append(
            AutomatedEvaluationRecord(
                run_id=f"pass-{index}",
                target_id="agent:research",
                mode="offline",
                rule_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.RULE_BASED,
                    passed=True,
                    score=1.0,
                ),
                llm_judge_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.LLM_JUDGE,
                    passed=True,
                    score=0.9,
                ),
                final_passed=True,
            )
        )
    for index in range(failed_records):
        records.append(
            AutomatedEvaluationRecord(
                run_id=f"fail-{index}",
                target_id="agent:research",
                mode="offline",
                rule_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.RULE_BASED,
                    passed=False,
                    score=0.0,
                ),
                llm_judge_result=AutomatedEvaluatorResult(
                    evaluator_type=EvaluatorType.LLM_JUDGE,
                    passed=False,
                    score=0.0,
                ),
                final_passed=False,
            )
        )
    return AutomatedEvaluationReport(records=records)


def test_evaluation_registry_trend_report_builds_release_comparison() -> None:
    trend_report = build_evaluation_registry_trend_report(
        snapshots=[
            EvaluationReleaseSnapshot(
                release_id="2026.05",
                automated_report=_automated_report(passed_records=1, failed_records=1),
            ),
            EvaluationReleaseSnapshot(
                release_id="2026.06",
                automated_report=_automated_report(passed_records=2, failed_records=0),
            ),
        ]
    )
    assert len(trend_report.comparisons) == 1
    comparison = trend_report.comparisons[0]
    assert comparison.release_from == "2026.05"
    assert comparison.release_to == "2026.06"
    assert comparison.delta > 0.0
