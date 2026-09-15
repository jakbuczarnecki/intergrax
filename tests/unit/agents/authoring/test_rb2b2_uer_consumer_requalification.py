# © Artur Czarnecki. All rights reserved.

"""RB-2B2 — behavioral re-proof for historical EXECUTION_RUNTIME-01…06 consumer/kernel gaps."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from intergrax.agents.authoring.acp_run import run_acp_session
from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_contract_meta import AgentRiskLevel
from intergrax.contracts.agent_run import AgentRunRequest, RequestIdentity
from intergrax.contracts.agent_run_enums import AgentRunErrorCode, SideEffectMode
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.execution_identity import mint_run_id
from intergrax.agents.authoring.base import IntergraxAgent
from intergrax.agents.authoring.acp_session_host import ACP_HOST_CONTEXT_KEY
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from pydantic import BaseModel
from tests.unit.agents.conftest import make_acp_host_context
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.org_policy import lab_strict_org_envelope

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_ACP_MINT_PATCH = (
    "intergrax.agents.authoring.acp_run.default_execution_identity_authority.mint_execution_identity"
)


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_FAIL_TOOL = ToolContract(
    tool_id="fail.tool",
    name="fail.tool",
    description="always fails",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.LOW,
)


class _ThrowingAgent(IntergraxAgent):
    contract_id = "throw-probe"
    capabilities = ("demo.throw",)
    agent_name = "Throw Probe"
    agent_description = "RB-2B2 exception containment probe"
    risk_level = AgentRiskLevel.LOW
    max_steps = 1

    async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
        raise RuntimeError("rb2b2-unhandled-domain-failure")


def _strict_profile() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile(
        execution_mode=ExecutionMode.STRICT,
        organizational=lab_strict_org_envelope(),
    )


@pytest.mark.asyncio
async def test_rb2b2_uer02_kernel_merged_state_survives_failed_tool_step() -> None:
    """UER-FIX-B: outcome_applied=False can still leave merged state_root (historical defect)."""

    async def _fail_invoke(**kwargs: object) -> object:
        raise RuntimeError("tool failed")

    from intergrax.agents.persistence.declarative_tool_executor import (
        CallableDeclarativeToolInvoker,
    )

    invoker = CallableDeclarativeToolInvoker({"fail.tool": _fail_invoke})
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-rb2b2-uer02",
        side_effect_mode=SideEffectMode.DECLARATIVE,
        policy_engine=PolicyEngine(),
        declarative_tool_invoker=invoker,
        allow_permissive_missing_policy=True,
        state_root={"acp.state": {"schema_version": "acp.state.v1", "_version": 0, "phase": "before"}},
    )
    step_ctx = AgentStepContext(
        step_index=0,
        side_effect_mode=SideEffectMode.DECLARATIVE,
    )
    outcome = StepOutcome.continue_with(
        {"phase": "after-merge-intent"},
        requested_actions=[{"tool_id": "fail.tool", "args": {}}],
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.outcome_applied is False
    assert record.error_code == AgentRunErrorCode.TOOL_FAILED
    blob = kernel_ctx.state_root.get("acp.state") or kernel_ctx.state_root
    assert blob.get("phase") == "after-merge-intent"


@pytest.mark.asyncio
async def test_rb2b2_uer03_acp_resume_still_mints_execution_identity() -> None:
    """UER-FIX-C: resume path re-enters mint before checkpoint attempt continuity exists."""

    class _OneStepAgent(IntergraxAgent):
        contract_id = "resume-probe"
        capabilities = ("demo.resume",)
        agent_name = "Resume Probe"
        agent_description = "RB-2B2 resume identity probe"
        risk_level = AgentRiskLevel.LOW
        max_steps = 1

        async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
            return StepOutcome.complete({"done": True})

    run_id = mint_run_id()
    request = AgentRunRequest(
        input="resume",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            "run_id": run_id,
            "resume": True,
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch(_ACP_MINT_PATCH) as mint:
        from intergrax.runtime.execution.identity_authority import RootTaskIdentity
        from intergrax.contracts.execution_identity import mint_attempt_id, mint_execution_id

        mint.return_value = RootTaskIdentity(
            run_id=run_id,
            attempt_id=mint_attempt_id(),
            execution_id=mint_execution_id(),
        )
        with patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"):
            await run_acp_session(_OneStepAgent(), request)
    mint.assert_called_once()


@pytest.mark.asyncio
async def test_rb2b2_uer04_unexpected_agent_exception_escapes_acp_session() -> None:
    """UER-FIX-D: non-budget agent exceptions are not converted to typed AgentRunResult."""

    request = AgentRunRequest(
        input="throw",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"):
        with pytest.raises(RuntimeError, match="rb2b2-unhandled-domain-failure"):
            await run_acp_session(_ThrowingAgent(), request)


@pytest.mark.asyncio
async def test_rb2b2_uer01_acp_session_uses_fresh_policy_engine_instance() -> None:
    """UER-FIX-A: direct ACP constructs its own PolicyEngine instead of host-carried engine."""

    constructed: list[PolicyEngine] = []

    class _RecordingPolicyEngine(PolicyEngine):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            constructed.append(self)

    class _OneStepAgent(IntergraxAgent):
        contract_id = "policy-probe"
        capabilities = ("demo.policy",)
        agent_name = "Policy Probe"
        agent_description = "RB-2B2 policy propagation probe"
        risk_level = AgentRiskLevel.LOW
        max_steps = 1

        async def on_next_step(self, step_ctx: AgentStepContext) -> StepOutcome:
            return StepOutcome.complete({"ok": True})

    request = AgentRunRequest(
        input="policy",
        identity=RequestIdentity(tenant_id="tenant-a", user_id="user-1"),
        metadata={
            ACP_HOST_CONTEXT_KEY: make_acp_host_context(_strict_profile()),
        },
    )
    with patch("intergrax.agents.authoring.acp_run.PolicyEngine", _RecordingPolicyEngine):
        with patch("intergrax.agents.authoring.acp_uaep_shim.attach_acp_catalog_exec_ctx"):
            await run_acp_session(_OneStepAgent(), request)
    assert len(constructed) >= 1
