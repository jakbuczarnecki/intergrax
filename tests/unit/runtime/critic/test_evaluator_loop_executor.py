# © Artur Czarnecki. All rights reserved.

"""CRIT-V-4.1 EvaluatorLoopExecutor tests."""

from __future__ import annotations

import pytest

from intergrax.runtime.critic.contracts import (
    CriticAction,
    CriticLayer,
    CriticScope,
    CriticVerdict,
    LayerVerdict,
)
from intergrax.runtime.critic.evaluator_loop_executor import (
    EvaluatorLoopDecision,
    EvaluatorLoopExecutor,
    EvaluatorLoopIterationState,
)
from intergrax.runtime.critic.evaluator_loop_spec import EvaluatorLoopSpec

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _failed_verdict(*, action: CriticAction = CriticAction.REVISE) -> CriticVerdict:
    return CriticVerdict(
        scope=CriticScope.NODE_PARTIAL,
        passed=False,
        layers=[
            LayerVerdict(
                layer=CriticLayer.L1_SEMANTIC,
                passed=False,
                errors=["below threshold"],
            ),
        ],
        recommended_action=action,
        failure_reasons=["below threshold"],
    )


def test_evaluator_loop_executor_revise_while_budget_remains() -> None:
    executor = EvaluatorLoopExecutor(
        spec=EvaluatorLoopSpec(max_iterations=2, revise_node_id="revise-1"),
    )
    outcome = executor.decide_after_verdict(
        _failed_verdict(),
        state=EvaluatorLoopIterationState(worker_node_id="worker-1", iteration=0),
        tenant_id="t1",
        task_id="task-1",
        agent_id="worker",
        node_id="worker-1",
    )
    assert outcome.decision is EvaluatorLoopDecision.REVISE
    assert outcome.revise_node_id == "revise-1"


def test_evaluator_loop_executor_fail_when_budget_exhausted() -> None:
    executor = EvaluatorLoopExecutor(
        spec=EvaluatorLoopSpec(max_iterations=2, revise_node_id="revise-1", escalate_on_exhaustion=False),
    )
    outcome = executor.decide_after_verdict(
        _failed_verdict(),
        state=EvaluatorLoopIterationState(worker_node_id="worker-1", iteration=1),
        tenant_id="t1",
        task_id="task-1",
        agent_id="worker",
    )
    assert outcome.decision is EvaluatorLoopDecision.FAIL


def test_evaluator_loop_executor_escalate_on_exhaustion() -> None:
    executor = EvaluatorLoopExecutor(
        spec=EvaluatorLoopSpec(max_iterations=2, revise_node_id="revise-1", escalate_on_exhaustion=True),
    )
    outcome = executor.decide_after_verdict(
        _failed_verdict(),
        state=EvaluatorLoopIterationState(worker_node_id="worker-1", iteration=1),
        tenant_id="t1",
        task_id="task-1",
        agent_id="worker",
    )
    assert outcome.decision is EvaluatorLoopDecision.ESCALATE_HITL
