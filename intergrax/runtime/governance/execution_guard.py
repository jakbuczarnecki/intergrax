# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from dataclasses import dataclass
from typing import List

from intergrax.logging import IntergraxLogging
from intergrax.runtime.governance.contracts.metrics_record_dto import RunMetricsRecord
from intergrax.runtime.governance.contracts.metrics_store import ExecutionMetricsStore
from intergrax.runtime.replay.service import ReplayService
from intergrax.runtime.replay.metrics import ExecutionMetrics, ExecutionMetricsEngine
from intergrax.runtime.replay.policy import ExecutionPolicyEngine, PolicyDecision
from intergrax.runtime.replay.regression import RegressionSignals
from intergrax.runtime.governance.history_evaluator import HistoryAwareEvaluator
from intergrax.runtime.governance.policy_actions import PolicyActionHandler


@dataclass(slots=True)
class GovernanceEvaluation:
    decision: PolicyDecision
    metrics: ExecutionMetrics
    regression: RegressionSignals
    

class ExecutionGuard:
    """
    Central runtime governance coordinator.

    Responsibilities:
    - reconstruct execution
    - compute metrics
    - load historical metrics
    - evaluate historical deviations
    - apply policy rules
    - execute policy actions
    """

    def __init__(
        self,
        replay_service: ReplayService,
        metrics_engine: ExecutionMetricsEngine,
        history_evaluator: HistoryAwareEvaluator,
        policy_engine: ExecutionPolicyEngine,
        metrics_store: ExecutionMetricsStore,
        actions: List[PolicyActionHandler],
        history_window: int = 20,
    ) -> None:
        self._replay = replay_service
        self._metrics_engine = metrics_engine
        self._history_eval = history_evaluator
        self._policy_engine = policy_engine
        self._metrics_store = metrics_store
        self._actions = actions
        self._history_window = history_window
        self._logger = IntergraxLogging.get_logger(__name__, component="governance")


    def evaluate_run(
        self,
        run_id: str,
        agent_id: str,
    ) -> GovernanceEvaluation:
        """
        Evaluates a completed run and enforces governance decisions.
        """

        # 1. Reconstruct execution
        reconstructed = self._replay.inspect_run(run_id)

        # 2. Compute metrics for current run
        metrics = self._metrics_engine.compute(reconstructed)

        # 3. Load historical metrics for this agent
        previous_records = self._metrics_store.get_recent(
            agent_id=agent_id,
            limit=self._history_window,
        )
        previous_metrics = [r.metrics for r in previous_records]

        # 4. Detect history-based regression signals
        regression: RegressionSignals = self._history_eval.evaluate(
            current=metrics,
            previous_runs=previous_metrics,
        )

        # 5. Evaluate policy
        decision: PolicyDecision = self._policy_engine.evaluate(
            metrics=metrics,
            regression=regression,
        )

        # 6. Emit governance log
        self._logger.info(
            "Execution governance decision",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "decision": decision.decision.value,
                "reasons": decision.reasons,
            },
        )

        # 7. Execute policy actions (may raise)
        for action in self._actions:
            action.handle(run_id, decision)

        # 8. Persist metrics only after successful enforcement
        record = RunMetricsRecord(
            run_id=run_id,
            agent_id=agent_id,
            metrics=metrics,
        )
        self._metrics_store.save(record)

        return GovernanceEvaluation(
            decision=decision,
            metrics=metrics,
            regression=regression,
        )

