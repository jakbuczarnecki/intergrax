# © Artur Czarnecki. All rights reserved.

"""Emit adaptive signals from Nexus and RuntimeEngine paths (Phase W-ADAPT-1.10–1.11)."""

from __future__ import annotations

from intergrax.runtime.architecture.online_evaluation_models import OnlineEvaluationObservation
from intergrax.runtime.adaptive.signal_collector import SignalAssemblyInput, SignalCollector
from intergrax.runtime.adaptive.contracts import HarnessOutcomeSignal
from intergrax.runtime.architecture.online_evaluation_registry import OnlineEvaluationRegistry
from intergrax.runtime.metrics.export import RunMetricsExport
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.task.task import Task, TaskResult, TaskState


def _resolve_task_class(task: Task, result: TaskResult) -> str:
    if task.classification:
        return task.classification
    capability = task.context.capability
    if capability:
        return str(capability)
    return result.agent_id or task.agent_id or "unknown"


def _resolve_hitl_interventions(task: Task, result: TaskResult) -> int:
    summary = result.summary
    if summary is None:
        return task.runtime.governance.escalation_level
    chain_len = len(summary.escalation_chain)
    level = summary.escalation_level
    return max(chain_len, level)


def _latest_evaluation_for_run(
    registry: OnlineEvaluationRegistry | None,
    run_id: str,
) -> OnlineEvaluationObservation | None:
    if registry is None:
        return None
    observations = registry.list_observations()
    matches = [item for item in observations if item.run_id == run_id]
    if not matches:
        return None
    return matches[-1]


def record_task_outcome_signal(
    collector: SignalCollector,
    *,
    task: Task,
    result: TaskResult,
    run_metrics: RunMetricsExport | None = None,
    regression: RegressionSignals | None = None,
    evaluation_registry: OnlineEvaluationRegistry | None = None,
    run_budget: RunBudget | None = None,
) -> HarnessOutcomeSignal | None:
    """Persist a signal when a Nexus task completes successfully."""
    if result.state != TaskState.COMPLETED:
        return None

    summary = result.summary
    actual_cost = summary.metrics.cost if summary is not None else None
    actual_tokens = summary.metrics.total_tokens if summary is not None else None
    validation_passed = True
    if summary is not None and summary.validation is not None:
        validation_passed = summary.validation.valid

    observation = _latest_evaluation_for_run(evaluation_registry, result.run_id)

    return collector.record(
        SignalAssemblyInput(
            run_id=result.run_id,
            tenant_id=task.tenant_id,
            application_id=collector.application_id,
            agent_id=result.agent_id or task.agent_id,
            task_class=_resolve_task_class(task, result),
            validation_passed=validation_passed,
            run_metrics=run_metrics,
            regression=regression,
            evaluation_observation=observation,
            run_budget=run_budget,
            actual_cost=actual_cost,
            actual_tokens=actual_tokens,
            hitl_interventions=_resolve_hitl_interventions(task, result),
        )
    )


def record_runtime_engine_outcome_signal(
    collector: SignalCollector,
    *,
    request: RuntimeRequest,
    run_id: str,
    answer: str,
    latency_ms: int,
    total_tokens: int,
    actual_cost: float | None = None,
    run_budget: RunBudget | None = None,
    evaluation_registry: OnlineEvaluationRegistry | None = None,
) -> HarnessOutcomeSignal:
    """Persist a signal for a non-Nexus RuntimeEngine run."""
    observation = _latest_evaluation_for_run(evaluation_registry, run_id)
    metrics = RunMetricsExport(
        run_id=run_id,
        tenant_id=request.tenant_id,
        agent_id=request.agent_id,
        duration_ms=latency_ms,
        event_count=0,
        cost=actual_cost,
        total_tokens=total_tokens,
    )
    return collector.record(
        SignalAssemblyInput(
            run_id=run_id,
            tenant_id=request.tenant_id,
            application_id=collector.application_id,
            agent_id=request.agent_id,
            task_class=str(request.metadata.get("capability") or request.agent_id),
            validation_passed=bool(answer.strip()),
            run_metrics=metrics,
            evaluation_observation=observation,
            run_budget=run_budget,
            actual_cost=actual_cost,
            actual_tokens=total_tokens,
        )
    )
