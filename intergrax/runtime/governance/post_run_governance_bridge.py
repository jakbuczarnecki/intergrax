# © Artur Czarnecki. All rights reserved.

"""POST_RUN governance lifecycle bridge (G3B)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from intergrax.logging import IntergraxLogging
from intergrax.runtime.governance.execution_guard import GovernanceEvaluation


@runtime_checkable
class PostRunGovernanceService(Protocol):
    def evaluate(self, run_id: str, agent_id: str) -> GovernanceEvaluation | None: ...


def invoke_post_run_governance(
    governance_service: PostRunGovernanceService | None,
    *,
    run_id: str,
    agent_id: str,
) -> GovernanceEvaluation | None:
    """Invoke post-run governance when a service is configured for the run."""
    if governance_service is None:
        return None
    if not (run_id or "").strip() or not (agent_id or "").strip():
        return None
    evaluation = governance_service.evaluate(run_id=run_id, agent_id=agent_id)
    IntergraxLogging.get_logger(__name__, component="governance").info(
        "Post-run governance invoked",
        extra={
            "run_id": run_id,
            "agent_id": agent_id,
            "evaluated": evaluation is not None,
        },
    )
    return evaluation
