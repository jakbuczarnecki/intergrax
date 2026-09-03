# © Artur Czarnecki. All rights reserved.

"""Evaluator-loop routing — critique→revise cycles (Phase CRIT-V-4.1)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from intergrax.runtime.critic.contracts import CriticAction, CriticVerdict
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec
from intergrax.runtime.critic.trace import CriticTraceEmitter


class EvaluatorLoopDecision(str, Enum):
    """Next orchestration step after a critic verdict inside an evaluator loop."""

    CONTINUE = "continue"
    REVISE = "revise"
    FAIL = "fail"
    ESCALATE_HITL = "escalate_hitl"


@dataclass(frozen=True, slots=True)
class EvaluatorLoopIterationState:
    """Mutable iteration counter for a worker node in an evaluator loop."""

    worker_node_id: str
    iteration: int = 0


@dataclass(frozen=True, slots=True)
class EvaluatorLoopOutcome:
    """Routing decision for graph executor after critic verification."""

    decision: EvaluatorLoopDecision
    iteration: int
    revise_node_id: str | None = None
    failure_reasons: tuple[str, ...] = ()


class EvaluatorLoopExecutor:
    """
    Bounded critique→revise router for ``CoordinationPattern.EVALUATOR_LOOP``.

    Does not execute agents — returns routing decisions for ``GraphExecutor``.
    """

    def __init__(
        self,
        *,
        spec: EvaluatorLoopSpec,
        trace_emitter: CriticTraceEmitter | None = None,
    ) -> None:
        self._spec = spec
        self._trace = trace_emitter

    @property
    def spec(self) -> EvaluatorLoopSpec:
        return self._spec

    def decide_after_verdict(
        self,
        verdict: CriticVerdict,
        *,
        state: EvaluatorLoopIterationState,
        tenant_id: str,
        task_id: str,
        agent_id: str,
        node_id: str | None = None,
    ) -> EvaluatorLoopOutcome:
        if verdict.passed:
            self._emit_iteration(
                tenant_id=tenant_id,
                task_id=task_id,
                agent_id=agent_id,
                node_id=node_id,
                iteration=state.iteration,
                passed=True,
            )
            return EvaluatorLoopOutcome(
                decision=EvaluatorLoopDecision.CONTINUE,
                iteration=state.iteration,
            )

        reasons = tuple(verdict.failure_reasons)
        self._emit_iteration(
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
            node_id=node_id,
            iteration=state.iteration,
            passed=False,
        )

        remaining = self._spec.max_iterations - state.iteration - 1
        revise_node_id = (self._spec.revise_node_id or "").strip() or None
        if (
            verdict.recommended_action in (CriticAction.REVISE, CriticAction.RETRY)
            and remaining > 0
            and revise_node_id is not None
        ):
            return EvaluatorLoopOutcome(
                decision=EvaluatorLoopDecision.REVISE,
                iteration=state.iteration,
                revise_node_id=revise_node_id,
                failure_reasons=reasons,
            )

        if self._spec.escalate_on_exhaustion:
            return EvaluatorLoopOutcome(
                decision=EvaluatorLoopDecision.ESCALATE_HITL,
                iteration=state.iteration,
                failure_reasons=reasons,
            )

        return EvaluatorLoopOutcome(
            decision=EvaluatorLoopDecision.FAIL,
            iteration=state.iteration,
            failure_reasons=reasons,
        )

    def bump_iteration(self, state: EvaluatorLoopIterationState) -> EvaluatorLoopIterationState:
        return EvaluatorLoopIterationState(
            worker_node_id=state.worker_node_id,
            iteration=state.iteration + 1,
        )

    def _emit_iteration(
        self,
        *,
        tenant_id: str,
        task_id: str,
        agent_id: str,
        node_id: str | None,
        iteration: int,
        passed: bool,
    ) -> None:
        if self._trace is None:
            return
        self._trace.emit_evaluator_loop(
            tenant_id=tenant_id,
            task_id=task_id,
            agent_id=agent_id,
            iteration=iteration,
            passed=passed,
            node_id=node_id,
        )
