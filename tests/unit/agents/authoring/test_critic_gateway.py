# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from intergrax.agents.authoring.critic_gateway import (
    ReflectionCriticOutcome,
    resolve_critic_hooks,
    verify_reflection_draft,
)
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.validation import ValidationResult
from intergrax.runtime.critic.contracts import CriticAction, CriticScope, CriticVerdict
from intergrax.runtime.critic.critic_wiring import CriticGraphHooks, CriticHookConfig

pytestmark = [pytest.mark.unit, pytest.mark.gate]


def _contract() -> AgentContract:
    return AgentContract(
        id="reflection_probe",
        name="Reflection Probe",
        description="test",
        capabilities=["harness.pattern.reflection"],
    )


def test_verify_reflection_draft_without_hooks_returns_none() -> None:
    step_ctx = AgentStepContext(
        run_id="run-1",
        agent_id="reflection_probe",
        metadata={AcpRunContextKey.TENANT_ID: "tenant-a"},
    )
    assert verify_reflection_draft(step_ctx, contract=_contract(), draft="hello") is None


def test_resolve_critic_hooks_from_metadata() -> None:
    hooks = MagicMock(spec=CriticGraphHooks)
    step_ctx = AgentStepContext(
        metadata={AcpRunContextKey.CRITIC_HOOKS: hooks},
    )
    assert resolve_critic_hooks(step_ctx) is hooks


@patch("intergrax.agents.authoring.critic_gateway.validate_uaep_step_with_critic_detail")
def test_verify_reflection_draft_maps_cvl_verdict(mock_validate) -> None:
    hooks = CriticGraphHooks(
        orchestrator=MagicMock(),
        config=CriticHookConfig(verify_uaep_step=True),
    )
    verdict = CriticVerdict(
        scope=CriticScope.UAEP_STEP,
        passed=False,
        recommended_action=CriticAction.REVISE,
        failure_reasons=["needs_more_citations"],
    )
    mock_validate.return_value = (ValidationResult(valid=False), verdict)
    step_ctx = AgentStepContext(
        run_id="run-2",
        agent_id="reflection_probe",
        metadata={
            AcpRunContextKey.CRITIC_HOOKS: hooks,
            AcpRunContextKey.TENANT_ID: "tenant-b",
        },
    )
    outcome = verify_reflection_draft(step_ctx, contract=_contract(), draft="draft text")
    assert isinstance(outcome, ReflectionCriticOutcome)
    assert outcome.action == CriticAction.REVISE
    assert outcome.passed is False
    mock_validate.assert_called_once()
