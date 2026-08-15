# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import re
from unittest.mock import patch

import pytest

from intergrax.agents.authoring.acp_run import run_acp_session
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.org_policy import lab_strict_org_envelope
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, AgentRunStatus, TerminalReason
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_run_id,
    mint_task_id,
    peek_active_execution_identity,
    validate_run_id,
    validate_task_id,
)
from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from testing_support.builder import FakeLLMAdapter, build_in_memory_session_manager
from tests.unit.agents.conftest import make_acp_host_context

_CANONICAL_ID = re.compile(r"^(task|run|attempt)_[0-9a-f]{32}$")

pytestmark = [pytest.mark.unit, pytest.mark.gate]


class _IdentityProbeAgent(IntergraxAgent):
    contract_id = "identity-probe"
    capabilities = ("demo.identity",)
    agent_name = "Identity Probe"
    agent_description = "ACP identity contract probe"
    risk_level = AgentRiskLevel.LOW
    max_steps = 3

    captured_active_identity: tuple[str, str] | None = None

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            production_mode=False,
            enable_rag=False,
            enable_websearch=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        _IdentityProbeAgent.captured_active_identity = peek_active_execution_identity()
        return StepOutcome.complete(
            output={"ok": True},
            terminal_reason=TerminalReason.GOAL_MET,
        )


class _WidenConfigureRunAgent(IntergraxAgent):
    contract_id = "strict_probe"
    capabilities = ("strict.probe",)
    agent_name = "Widen Configure Run Agent"

    def build_context(self, request: RuntimeRequest) -> RuntimeContext:
        config = RuntimeConfig(
            llm_adapter=FakeLLMAdapter(),
            enable_rag=False,
            production_mode=False,
        )
        return RuntimeContext.build(
            config=config,
            session_manager=build_in_memory_session_manager(),
        )

    def configure_run(self, merged: object) -> dict[str, object]:
        _ = merged
        return {"allowed_tools": ["evil.tool"]}


def _request(**metadata: object) -> AgentRunRequest:
    return AgentRunRequest(
        input="probe",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata=dict(metadata),
    )


def _strict_profile() -> ApplicationEnvironmentProfile:
    profile = ApplicationEnvironmentProfile.lab_defaults(profile_id="strict.host")
    return profile.model_copy(
        update={
            "execution_mode": ExecutionMode.STRICT,
            "organizational_policy": lab_strict_org_envelope(),
        },
    )


@pytest.mark.asyncio
async def test_acp_mints_identity_once_when_absent() -> None:
    minted_run = mint_run_id()
    minted_task = mint_task_id()
    agent = _IdentityProbeAgent()
    with (
        patch("intergrax.agents.authoring.acp_run.mint_run_id", return_value=minted_run) as mint_run,
        patch("intergrax.agents.authoring.acp_run.mint_task_id", return_value=minted_task) as mint_task,
        patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"),
    ):
        result = await run_acp_session(agent, _request())
    assert mint_run.call_count == 1
    assert mint_task.call_count == 1
    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.run_id == minted_run
    assert _CANONICAL_ID.fullmatch(result.run_id)


@pytest.mark.asyncio
async def test_acp_preserves_supplied_canonical_identity() -> None:
    supplied_run = mint_run_id()
    supplied_task = mint_task_id()
    agent = _IdentityProbeAgent()
    with (
        patch("intergrax.agents.authoring.acp_run.mint_run_id") as mint_run,
        patch("intergrax.agents.authoring.acp_run.mint_task_id") as mint_task,
        patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"),
    ):
        result = await run_acp_session(
            agent,
            _request(run_id=supplied_run, task_id=supplied_task),
        )
    mint_run.assert_not_called()
    mint_task.assert_not_called()
    assert result.status == AgentRunStatus.SUCCEEDED
    assert result.run_id == supplied_run


@pytest.mark.asyncio
async def test_acp_malformed_run_id_fails_without_replacement() -> None:
    malformed_run = "run_not_canonical"
    agent = _IdentityProbeAgent()
    _IdentityProbeAgent.captured_active_identity = None
    with patch("intergrax.agents.authoring.acp_run.mint_run_id") as mint_run:
        result = await run_acp_session(agent, _request(run_id=malformed_run))
    mint_run.assert_not_called()
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.ERROR
    assert result.run_id == ""
    assert result.trace.run_id == ""
    assert result.errors[0].code == AgentRunErrorCode.INTERNAL_ERROR
    assert "malformed execution identity" in result.errors[0].message
    assert malformed_run not in result.run_id
    assert _IdentityProbeAgent.captured_active_identity is None


@pytest.mark.asyncio
async def test_acp_malformed_task_id_fails_without_replacement() -> None:
    malformed_task = "task_not_canonical"
    supplied_run = mint_run_id()
    agent = _IdentityProbeAgent()
    _IdentityProbeAgent.captured_active_identity = None
    with patch("intergrax.agents.authoring.acp_run.mint_task_id") as mint_task:
        result = await run_acp_session(
            agent,
            _request(run_id=supplied_run, task_id=malformed_task),
        )
    mint_task.assert_not_called()
    assert result.status == AgentRunStatus.FAILED
    assert result.run_id == ""
    assert result.trace.run_id == ""
    assert malformed_task not in result.run_id
    assert result.run_id != supplied_run
    assert _IdentityProbeAgent.captured_active_identity is None


@pytest.mark.asyncio
async def test_configure_run_strict_violation_uses_boundary_identity_without_fallback() -> None:
    supplied_run = mint_run_id()
    agent = _WidenConfigureRunAgent()
    request = AgentRunRequest(
        input="strict-deny",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            "run_id": supplied_run,
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch("intergrax.agents.authoring.acp_run.mint_run_id") as mint_run:
        result = await run_acp_session(agent, request)
    mint_run.assert_not_called()
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.POLICY_DENIED
    assert result.run_id == supplied_run
    assert validate_run_id(result.run_id) == supplied_run


@pytest.mark.asyncio
async def test_configure_run_strict_violation_mints_once_when_identity_absent() -> None:
    minted_run = mint_run_id()
    agent = _WidenConfigureRunAgent()
    request = AgentRunRequest(
        input="strict-deny",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch("intergrax.agents.authoring.acp_run.mint_run_id", return_value=minted_run) as mint_run:
        result = await run_acp_session(agent, request)
    assert mint_run.call_count == 1
    assert result.status == AgentRunStatus.FAILED
    assert result.run_id == minted_run


@pytest.mark.asyncio
async def test_configure_run_strict_violation_does_not_mask_malformed_run_id() -> None:
    malformed_run = "run_bad"
    agent = _WidenConfigureRunAgent()
    request = AgentRunRequest(
        input="strict-deny",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            "run_id": malformed_run,
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch("intergrax.agents.authoring.acp_run.mint_run_id") as mint_run:
        result = await run_acp_session(agent, request)
    mint_run.assert_not_called()
    assert result.status == AgentRunStatus.FAILED
    assert result.terminal_reason == TerminalReason.ERROR
    assert result.run_id == ""
    assert malformed_run not in result.run_id


@pytest.mark.asyncio
async def test_acp_binds_and_resets_active_execution_identity() -> None:
    supplied_run = mint_run_id()
    supplied_task = mint_task_id()
    attempt_id = mint_attempt_id()
    agent = _IdentityProbeAgent()
    _IdentityProbeAgent.captured_active_identity = None
    with (
        patch("intergrax.agents.authoring.acp_run.mint_attempt_id", return_value=attempt_id),
        patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"),
    ):
        result = await run_acp_session(
            agent,
            _request(run_id=supplied_run, task_id=supplied_task),
        )
    assert result.status == AgentRunStatus.SUCCEEDED
    assert _IdentityProbeAgent.captured_active_identity == (supplied_run, attempt_id)
    assert peek_active_execution_identity() is None
    assert validate_run_id(result.run_id) == supplied_run
    assert validate_task_id(supplied_task)
