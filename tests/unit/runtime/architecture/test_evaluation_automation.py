from __future__ import annotations

from intergrax.runtime.architecture.evaluation_automation import (
    EvaluationSignal,
    evaluate_automated_results,
)
from intergrax.runtime.architecture.evaluation_modes import (
    EvaluationMode,
    EvaluationModeResult,
)


def test_automated_evaluation_passes_when_rule_and_llm_scores_are_healthy() -> None:
    report = evaluate_automated_results(
        mode_results=[
            EvaluationModeResult(
                run_id="run-1",
                target_id="agent:research",
                mode=EvaluationMode.OFFLINE,
                success=True,
                score=0.9,
            )
        ],
        rule_signals_by_run_id={
            "run-1": [
                EvaluationSignal(signal_id="policy", value=1.0, threshold=1.0),
                EvaluationSignal(signal_id="quality", value=0.9, threshold=0.8),
            ]
        },
        llm_judge_scores_by_run_id={"run-1": 0.85},
    )
    assert report.records[0].final_passed is True


def test_automated_evaluation_fails_when_llm_judge_below_threshold() -> None:
    report = evaluate_automated_results(
        mode_results=[
            EvaluationModeResult(
                run_id="run-2",
                target_id="agent:research",
                mode=EvaluationMode.ONLINE,
                success=True,
                score=0.9,
            )
        ],
        rule_signals_by_run_id={
            "run-2": [EvaluationSignal(signal_id="policy", value=1.0, threshold=1.0)]
        },
        llm_judge_scores_by_run_id={"run-2": 0.5},
    )
    assert report.records[0].final_passed is False
    assert report.records[0].llm_judge_result.reasons
