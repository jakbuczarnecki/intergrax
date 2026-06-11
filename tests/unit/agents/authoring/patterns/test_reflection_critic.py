# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.agents.authoring.patterns.reference import PatternReflectionProbe
from intergrax.agents.authoring.patterns.states import ReflectionSessionState
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.runtime.critic.contracts import CriticAction
from intergrax.agents.authoring.critic_gateway import ReflectionCriticOutcome

pytestmark = [pytest.mark.unit, pytest.mark.gate]


@pytest.mark.asyncio
@patch("intergrax.agents.authoring.patterns.reflection.verify_reflection_draft")
async def test_reflection_agent_cvl_pass_completes_early(mock_verify) -> None:
    mock_verify.return_value = ReflectionCriticOutcome(
        passed=True,
        action=CriticAction.CONTINUE,
        summary="ok",
        verdict=MagicMock(),
    )
    agent = PatternReflectionProbe()
    state = ReflectionSessionState(phase="critique", draft="final draft")
    step_ctx = AgentStepContext(
        run_id="run-ref",
        agent_id=agent.contract_id,
        metadata={AcpRunContextKey.CRITIC_HOOKS: MagicMock()},
        state_snapshot={ACP_STATE_KEY: state.model_dump(by_alias=True)},
    )
    outcome = await agent.on_next_step(step_ctx)
    assert outcome.is_terminal
    assert outcome.terminal_reason is not None
    mock_verify.assert_called_once()
