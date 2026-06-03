# © Artur Czarnecki. All rights reserved.

"""Online and shadow evaluation hooks for harness registry trends (Phase W-OPS.11)."""

from __future__ import annotations

from intergrax.runtime.architecture.evaluation_automation import (
    AutomatedEvaluationReport,
    EvaluationSignal,
    evaluate_automated_results,
)
from intergrax.runtime.architecture.evaluation_modes import EvaluationMode, EvaluationModeResult
from intergrax.runtime.architecture.evaluation_registry_trends import (
    EvaluationReleaseSnapshot,
    build_evaluation_registry_trend_report,
)
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationBatch,
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.architecture.online_evaluation_registry import (
    OnlineEvaluationRegistry,
    default_online_evaluation_registry,
)

__all__ = [
    "OnlineEvaluationBatch",
    "OnlineEvaluationMode",
    "OnlineEvaluationObservation",
    "append_online_evaluation_to_trend",
    "observations_to_automated_report",
    "record_shadow_observation",
]


def observations_to_automated_report(batch: OnlineEvaluationBatch) -> AutomatedEvaluationReport:
    """Convert harness online/shadow observations into V-EVAL automated report shape."""
    mode_results: list[EvaluationModeResult] = []
    rule_signals: dict[str, list[EvaluationSignal]] = {}
    llm_scores: dict[str, float] = {}
    for item in batch.observations:
        mode = (
            EvaluationMode.SHADOW
            if item.mode == OnlineEvaluationMode.SHADOW
            else EvaluationMode.ONLINE
        )
        mode_results.append(
            EvaluationModeResult(
                run_id=item.run_id,
                target_id=item.agent_id,
                mode=mode,
                success=item.passed,
                score=item.score,
                evidence_refs=[item.scenario_id],
            )
        )
        rule_signals[item.run_id] = [
            EvaluationSignal(
                signal_id=f"{item.scenario_id}:score",
                value=item.score,
                threshold=0.5,
            )
        ]
        llm_scores[item.run_id] = item.score
    return evaluate_automated_results(
        mode_results=mode_results,
        rule_signals_by_run_id=rule_signals,
        llm_judge_scores_by_run_id=llm_scores,
    )


def append_online_evaluation_to_trend(
    *,
    existing_snapshots: list[EvaluationReleaseSnapshot],
    batch: OnlineEvaluationBatch,
) -> tuple[EvaluationReleaseSnapshot, list]:
    """Append one release snapshot and rebuild trend comparisons."""
    report = observations_to_automated_report(batch)
    snapshot = EvaluationReleaseSnapshot(
        release_id=batch.release_id,
        automated_report=report,
    )
    updated = [*existing_snapshots, snapshot]
    trend = build_evaluation_registry_trend_report(updated)
    return snapshot, trend.comparisons


def record_shadow_observation(
    *,
    run_id: str,
    agent_id: str,
    scenario_id: str,
    passed: bool,
    score: float,
    registry: OnlineEvaluationRegistry | None = None,
) -> OnlineEvaluationObservation:
    """Record a single shadow-mode harness observation and append to the registry."""
    observation = OnlineEvaluationObservation(
        observation_id=f"shadow:{run_id}:{scenario_id}",
        run_id=run_id,
        agent_id=agent_id,
        mode=OnlineEvaluationMode.SHADOW,
        scenario_id=scenario_id,
        passed=passed,
        score=score,
    )
    target_registry = registry or default_online_evaluation_registry()
    target_registry.append(observation)
    return observation
