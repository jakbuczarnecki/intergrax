# © Artur Czarnecki. All rights reserved.

"""Assemble harness outcome signals from runtime sources (Phase W-ADAPT-1.4–1.9)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.runtime.adaptive.contracts import (
    HarnessOutcomeSignal,
    OutcomeEvalMode,
    UtilityWeights,
)
from intergrax.runtime.adaptive.cost_normalization import normalize_cost_against_budget
from intergrax.runtime.adaptive.llm_call_summary import LLMCallSummary
from intergrax.runtime.adaptive.signal_store import SignalStore
from intergrax.runtime.adaptive.utility import compute_utility
from intergrax.runtime.architecture.online_evaluation_models import (
    OnlineEvaluationMode,
    OnlineEvaluationObservation,
)
from intergrax.runtime.metrics.export import RunMetricsExport
from intergrax.runtime.nexus.budget.budget_models import RunBudget
from intergrax.runtime.replay.regression import RegressionSignals


def regression_flags_from_signals(regression: RegressionSignals | None) -> list[str]:
    """Translate regression dataclass fields into stable flag identifiers."""
    if regression is None:
        return []
    flags: list[str] = []
    if regression.step_explosion:
        flags.append("step_explosion")
    if regression.llm_cost_spike:
        flags.append("llm_cost_spike")
    if regression.tool_usage_drop:
        flags.append("tool_usage_drop")
    if regression.final_answer_changed:
        flags.append("final_answer_changed")
    return flags


def eval_mode_from_observation(
    observation: OnlineEvaluationObservation | None,
) -> OutcomeEvalMode:
    if observation is None:
        return OutcomeEvalMode.OFFLINE
    if observation.mode == OnlineEvaluationMode.SHADOW:
        return OutcomeEvalMode.SHADOW
    return OutcomeEvalMode.ONLINE


@dataclass(frozen=True, slots=True)
class SignalAssemblyInput:
    """Typed inputs for assembling a harness outcome signal."""

    run_id: str
    tenant_id: str
    application_id: str
    agent_id: str
    task_class: str
    validation_passed: bool = True
    run_metrics: RunMetricsExport | None = None
    regression: RegressionSignals | None = None
    evaluation_observation: OnlineEvaluationObservation | None = None
    run_budget: RunBudget | None = None
    actual_cost: float | None = None
    actual_tokens: int | None = None
    hitl_interventions: int = 0
    business_outcome: float | None = None
    last_llm_call: LLMCallSummary | None = None


class SignalCollector:
    """Collects and persists harness outcome signals (L4-O)."""

    def __init__(
        self,
        store: SignalStore,
        *,
        utility_weights: UtilityWeights | None = None,
        application_id: str = "harness.default",
    ) -> None:
        self._store = store
        self._utility_weights = utility_weights or UtilityWeights()
        self._application_id = application_id

    @property
    def store(self) -> SignalStore:
        return self._store

    @property
    def utility_weights(self) -> UtilityWeights:
        return self._utility_weights

    @property
    def application_id(self) -> str:
        return self._application_id

    def assemble_signal(self, assembly: SignalAssemblyInput) -> HarnessOutcomeSignal:
        """Build a signal from typed runtime inputs without persisting."""
        metrics = assembly.run_metrics
        behavioral = metrics.behavioral if metrics is not None else None

        latency_ms = metrics.duration_ms if metrics is not None else 0
        total_tokens = assembly.actual_tokens
        if total_tokens is None and metrics is not None:
            total_tokens = metrics.total_tokens or 0
        total_tokens = total_tokens or 0

        step_count = behavioral.step_count if behavioral is not None else 0
        tool_calls = behavioral.total_tool_calls if behavioral is not None else 0
        llm_calls = behavioral.total_llm_calls if behavioral is not None else 0

        actual_cost = assembly.actual_cost
        if actual_cost is None and metrics is not None:
            actual_cost = metrics.cost

        cost_normalized = normalize_cost_against_budget(
            total_tokens=total_tokens,
            actual_cost=actual_cost,
            run_budget=assembly.run_budget,
        )

        observation = assembly.evaluation_observation
        quality_score = observation.score if observation is not None else (
            1.0 if assembly.validation_passed else 0.0
        )
        eval_mode = eval_mode_from_observation(observation)
        regression_flags = regression_flags_from_signals(assembly.regression)
        last_llm = assembly.last_llm_call

        signal = HarnessOutcomeSignal(
            run_id=assembly.run_id,
            tenant_id=assembly.tenant_id,
            application_id=assembly.application_id or self._application_id,
            agent_id=assembly.agent_id,
            task_class=assembly.task_class,
            quality_score=quality_score,
            validation_passed=assembly.validation_passed,
            eval_mode=eval_mode,
            cost_normalized=cost_normalized,
            latency_ms=latency_ms,
            total_tokens=total_tokens,
            step_count=step_count,
            tool_calls=tool_calls,
            llm_calls=llm_calls,
            hitl_interventions=assembly.hitl_interventions,
            regression_flags=regression_flags,
            business_outcome=assembly.business_outcome,
            last_llm_finish_reason=last_llm.finish_reason if last_llm else None,
            last_llm_model=last_llm.model if last_llm else None,
            last_llm_provider=last_llm.provider if last_llm else None,
            last_llm_input_tokens=last_llm.input_tokens if last_llm else None,
            last_llm_output_tokens=last_llm.output_tokens if last_llm else None,
            last_llm_has_refusal=last_llm.has_refusal if last_llm else None,
            last_llm_has_tool_calls=last_llm.has_tool_calls if last_llm else None,
        )
        utility = compute_utility(
            quality_score=signal.quality_score,
            cost_normalized=signal.cost_normalized,
            latency_ms=signal.latency_ms,
            hitl_interventions=signal.hitl_interventions,
            regression_flags=signal.regression_flags,
            business_outcome=signal.business_outcome,
            weights=self._utility_weights,
        )
        return signal.model_copy(update={"utility": utility})

    def record(self, assembly: SignalAssemblyInput) -> HarnessOutcomeSignal:
        """Assemble and persist a harness outcome signal."""
        signal = self.assemble_signal(assembly)
        self._store.append(signal)
        return signal
