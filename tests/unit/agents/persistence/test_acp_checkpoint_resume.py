# © Artur Czarnecki. All rights reserved.

"""ACP-PROD-1 — typed session checkpoint resume integration."""

from __future__ import annotations

import pytest

from intergrax.agents.authoring.patterns.reference import PatternPlanExecuteProbe
from intergrax.agents.authoring.patterns.states import PlanExecuteSessionState
from intergrax.agents.persistence.checkpoint_store import InMemoryAgentCheckpointStore
from intergrax.agents.persistence.checkpoint_wiring import wire_acp_run_request
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunStatus
from intergrax.contracts.agent_run import AgentExecutionOptions


class _CountablePlanProbe(PatternPlanExecuteProbe):
    perceive_calls: int = 0

    async def perceive(self, step_ctx):  # type: ignore[no-untyped-def]
        _CountablePlanProbe.perceive_calls += 1
        return await super().perceive(step_ctx)


@pytest.mark.unit
@pytest.mark.gate
async def test_acp_checkpoint_resume_continues_plan_execute_phase() -> None:
    _CountablePlanProbe.perceive_calls = 0
    agent = _CountablePlanProbe()
    store = InMemoryAgentCheckpointStore()
    run_id = "acp-resume-run-1"

    base = AgentRunRequest(
        input="checkpoint-resume",
        identity=RequestIdentity(tenant_id="t-resume", user_id="u-resume"),
        metadata={"run_id": run_id, "user_id": "u-resume"},
    )

    partial = wire_acp_run_request(
        base.model_copy(
            update={
                "execution_options": AgentExecutionOptions(
                    max_steps=1,
                    checkpoint_every_step=True,
                ),
            },
        ),
        store,
        resume=False,
    )
    partial_result = await agent.run(partial)
    checkpoint = store.get_latest(run_id, "t-resume")
    assert checkpoint is not None
    assert checkpoint.step_index == 0
    assert _CountablePlanProbe.perceive_calls == 1
    phase_after_partial = checkpoint.state_root["acp.state.v1"]["phase"]
    assert phase_after_partial == "execute"

    resumed = wire_acp_run_request(
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
    )
    final_result = await agent.run(resumed)
    assert final_result.status == AgentRunStatus.SUCCEEDED
    assert _CountablePlanProbe.perceive_calls == 3
    final_phase = PlanExecuteSessionState.model_validate(
        final_result.state["acp.state.v1"]
    ).phase
    assert final_phase in {"synthesize", "done"}
