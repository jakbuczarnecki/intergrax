# © Artur Czarnecki. All rights reserved.

"""Automated evaluator contracts (rule-based and LLM-judge) for Phase V-EVAL.3."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field

from intergrax.runtime.architecture.evaluation_modes import EvaluationModeResult


class EvaluatorType(str, Enum):
    RULE_BASED = "rule_based"
    LLM_JUDGE = "llm_judge"


class EvaluationSignal(BaseModel):
    signal_id: str
    value: float
    threshold: float


class AutomatedEvaluatorResult(BaseModel):
    evaluator_type: EvaluatorType
    passed: bool
    score: float
    reasons: list[str] = Field(default_factory=list)


class AutomatedEvaluationRecord(BaseModel):
    run_id: str
    target_id: str
    mode: str
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    rule_result: AutomatedEvaluatorResult
    llm_judge_result: AutomatedEvaluatorResult
    final_passed: bool


class AutomatedEvaluationReport(BaseModel):
    schema_version: str = "1.0.0"
    records: list[AutomatedEvaluationRecord] = Field(default_factory=list)


def evaluate_automated_results(
    *,
    mode_results: list[EvaluationModeResult],
    rule_signals_by_run_id: dict[str, list[EvaluationSignal]],
    llm_judge_scores_by_run_id: dict[str, float],
) -> AutomatedEvaluationReport:
    records: list[AutomatedEvaluationRecord] = []
    for result in mode_results:
        signals = rule_signals_by_run_id.get(result.run_id, [])
        rule_result = _evaluate_rule_signals(signals)
        llm_score = llm_judge_scores_by_run_id.get(result.run_id, 0.0)
        llm_result = _evaluate_llm_judge(llm_score)
        records.append(
            AutomatedEvaluationRecord(
                run_id=result.run_id,
                target_id=result.target_id,
                mode=result.mode.value,
                rule_result=rule_result,
                llm_judge_result=llm_result,
                final_passed=rule_result.passed and llm_result.passed,
            )
        )
    return AutomatedEvaluationReport(records=records)


def _evaluate_rule_signals(signals: list[EvaluationSignal]) -> AutomatedEvaluatorResult:
    if not signals:
        return AutomatedEvaluatorResult(
            evaluator_type=EvaluatorType.RULE_BASED,
            passed=False,
            score=0.0,
            reasons=["No rule signals provided"],
        )
    passed_signals = [signal for signal in signals if signal.value >= signal.threshold]
    score = float(len(passed_signals)) / float(len(signals))
    reasons = [
        f"Signal below threshold: {signal.signal_id}"
        for signal in signals
        if signal.value < signal.threshold
    ]
    return AutomatedEvaluatorResult(
        evaluator_type=EvaluatorType.RULE_BASED,
        passed=score >= 0.8,
        score=score,
        reasons=reasons,
    )


def _evaluate_llm_judge(score: float) -> AutomatedEvaluatorResult:
    passed = score >= 0.75
    reasons = [] if passed else ["LLM judge score below threshold"]
    return AutomatedEvaluatorResult(
        evaluator_type=EvaluatorType.LLM_JUDGE,
        passed=passed,
        score=score,
        reasons=reasons,
    )
