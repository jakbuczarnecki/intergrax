# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from intergrax.logging import IntergraxLogging
from intergrax.runtime.governance.execution_guard import ExecutionGuard, GovernanceEvaluation


@dataclass(slots=True)
class GovernanceService:
    """
    Entry point for governance evaluation after run completion.
    """

    guard: ExecutionGuard
    _logger: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._logger = IntergraxLogging.get_logger(__name__, component="governance")

    def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation:
        evaluation = self.guard.evaluate_run(
            run_id=run_id,
            agent_id=agent_id,
        )

        self._logger.info(
            "Governance evaluation finished",
            extra={
                "run_id": run_id,
                "agent_id": agent_id,
                "decision": evaluation.decision.decision.value,
            },
        )

        return evaluation
