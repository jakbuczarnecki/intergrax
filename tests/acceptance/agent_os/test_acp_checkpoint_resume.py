# © Artur Czarnecki. All rights reserved.

"""
ACP-PROD-1 acceptance — typed ACP checkpoint resume (agent_os suite).

Exercises crash-after-one-step + resume on PlanExecute harness probe (mock LLM).
"""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns.reference import PatternPlanExecuteProbe
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.agents.persistence.checkpoint_wiring import wire_acp_run_request
from intergrax.contracts.agent_run import AgentExecutionOptions, AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus

pytestmark = [pytest.mark.integration, pytest.mark.agent_os, pytest.mark.gate]


class _CheckpointPlanProbe(PatternPlanExecuteProbe):
    perceive_calls: int = 0

    async def perceive(self, step_ctx):  # type: ignore[no-untyped-def]
        _CheckpointPlanProbe.perceive_calls += 1
        return await super().perceive(step_ctx)


@pytest.mark.asyncio
async def test_acceptance_05c_acp_checkpoint_resume() -> None:
    _CheckpointPlanProbe.perceive_calls = 0
    agent = _CheckpointPlanProbe()
    store = InMemoryAgentCheckpointStore()
    run_id = "acceptance-acp-ckpt-1"

    base = AgentRunRequest(
        input="acp-checkpoint-acceptance",
        identity=RequestIdentity(tenant_id="t-agent-os", user_id="u-acp"),
        metadata={"run_id": run_id, "user_id": "u-acp"},
    )

    await agent.run(
        wire_acp_run_request(
            base.model_copy(
                update={
                    "execution_options": AgentExecutionOptions(
                        max_steps=1,
                        checkpoint_every_step=True,
                    ),
                },
            ),
            store,
        ),
    )
    assert store.get_latest(run_id, "t-agent-os") is not None
    assert _CheckpointPlanProbe.perceive_calls == 1

    result = await agent.run(
        wire_acp_run_request(
            base.model_copy(
                update={
                    "execution_options": AgentExecutionOptions(
                        max_steps=10,
                        checkpoint_every_step=True,
                    ),
                },
            ),
            store,
            resume=True,
        ),
    )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert _CheckpointPlanProbe.perceive_calls == 3
