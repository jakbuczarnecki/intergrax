# © Artur Czarnecki. All rights reserved.

import pytest

from intergrax.agents.authoring.step_outcome import StepOutcome
from intergrax.contracts.agent_run_enums import (
    AgentRunErrorCode,
    SideEffectMode,
    TerminalReason,
)
from intergrax.tools.core.contracts import ToolContract, ToolRiskLevel
from intergrax.tools.tool_execution_profile import (
    ToolExecutionProfile,
    ToolMutability,
    ToolReversibility,
    build_profile_map,
)
from pydantic import BaseModel


class _In(BaseModel):
    pass


class _Out(BaseModel):
    pass


_MUTATING_TOOL = ToolContract(
    tool_id="email.send",
    name="email.send",
    description="send",
    input_schema=_In,
    output_schema=_Out,
    error_mapping={},
    side_effects=True,
    risk_level=ToolRiskLevel.HIGH,
)
from intergrax.applications.contracts.org_policy import (
    ChannelPolicy,
    OrganizationalPolicyContext,
)
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.runtime_policy import PolicyAction
from intergrax.agents.persistence.declarative_tool_executor import (
    CallableDeclarativeToolInvoker,
    DeclarativeToolInvokeResult,
)
from intergrax.agents.persistence.idempotency_keys import build_default_idempotency_key
from intergrax.agents.persistence.side_effect_ledger import SideEffectLedger
from intergrax.contracts.side_effect import SideEffectKind, SideEffectStatus
from intergrax.runtime.kernel.step_kernel import HarnessKernel, StepKernelContext
from intergrax.runtime.policy.policy_engine import PolicyEngine


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_policy_pre_deny() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        metadata={"policy_pre_deny": True},
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-1",
        policy_engine=PolicyEngine(),
    )
    outcome = StepOutcome.continue_with({"phase": "plan"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert record.policy_pre is not None
    assert record.policy_pre.action == PolicyAction.DENY
    assert len(kernel_ctx.events) >= 1


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_budget_exceeded() -> None:
    step_ctx = AgentStepContext(step_index=2)
    kernel_ctx = StepKernelContext(agent_id="demo", run_id="run-1", max_steps=2)
    outcome = StepOutcome.continue_with({"phase": "execute"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.budget_exceeded is True
    assert record.error_code == AgentRunErrorCode.MAX_STEPS_EXCEEDED


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_merges_state_and_appends_trace() -> None:
    step_ctx = AgentStepContext(step_index=0)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-trace",
        policy_engine=PolicyEngine(),
        state_root={"acp.state.v1": {"schema_version": "acp.state.v1", "_version": 0}},
    )
    outcome = StepOutcome.continue_with({"phase": "execute"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.outcome_applied is True
    assert record.state_version == 1
    assert kernel_ctx.run_trace.steps
    assert step_ctx.state_snapshot["acp.state.v1"]["phase"] == "execute"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_rejects_mixed_side_effect_mode() -> None:
    step_ctx = AgentStepContext(step_index=0, side_effect_mode=SideEffectMode.IMMEDIATE)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        side_effect_mode=SideEffectMode.IMMEDIATE,
    )
    outcome = StepOutcome.continue_with(
        {},
        diagnostics=None,
    )
    outcome = outcome.model_copy(update={"requested_actions": [{"tool_id": "demo.tool"}]})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.side_effect_mode_violation is True
    assert record.error_code == AgentRunErrorCode.VALIDATION_FAILED


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_org_policy_denies_channel() -> None:
    org = OrganizationalPolicyContext(
        organization_id="lab.virtual_org",
        channel_policy=ChannelPolicy(
            allowed_channels=["chat"],
            denied_channels=["phone"],
        ),
    )
    step_ctx = AgentStepContext(step_index=0, metadata={"channel": "phone"})
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-org",
        policy_engine=PolicyEngine(),
        organizational=org,
    )
    outcome = StepOutcome.continue_with({"phase": "plan"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert record.policy_pre is not None
    assert record.policy_pre.policy_rule_id == "org.channel.denied"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_org_policy_allows_happy_path_channel() -> None:
    org = OrganizationalPolicyContext(
        organization_id="lab.virtual_org",
        channel_policy=ChannelPolicy(
            allowed_channels=["chat", "ticket"],
            denied_channels=["phone"],
        ),
    )
    step_ctx = AgentStepContext(step_index=0, metadata={"channel": "chat"})
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-org-ok",
        policy_engine=PolicyEngine(),
        organizational=org,
    )
    outcome = StepOutcome.continue_with({"phase": "plan"})
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code is None
    assert record.outcome_applied is True
    assert record.step_record is not None
    assert not any(
        v.policy_rule_id.startswith("org.")
        and v.action == PolicyAction.DENY
        for v in record.step_record.policy_verdicts
    )


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_rejects_mutating_tool_without_idempotency_key() -> None:
    step_ctx = AgentStepContext(
        step_index=0,
        side_effect_mode=SideEffectMode.DECLARATIVE,
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-mutating",
        side_effect_mode=SideEffectMode.DECLARATIVE,
        policy_engine=PolicyEngine(),
        tool_profiles=build_profile_map([_MUTATING_TOOL]),
    )
    outcome = StepOutcome.continue_with({"phase": "send"})
    outcome = outcome.model_copy(
        update={"requested_actions": [{"tool_id": "email.send", "args": {"to": "x"}}]},
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.VALIDATION_FAILED
    assert record.step_record is not None
    assert record.step_record.diagnostics.get("tool_validation") == "acp.tool.idempotency_required"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_executes_declarative_actions_and_commits_ledger() -> None:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-decl",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target="email.send",
        args={"to": "x"},
    )
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success", external_ref="msg-kernel")

    step_ctx = AgentStepContext(
        step_index=0,
        side_effect_mode=SideEffectMode.DECLARATIVE,
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-decl",
        side_effect_mode=SideEffectMode.DECLARATIVE,
        policy_engine=PolicyEngine(),
        tool_profiles=build_profile_map([_MUTATING_TOOL]),
        side_effect_ledger=ledger,
        declarative_tool_invoker=CallableDeclarativeToolInvoker(_invoke),
    )
    outcome = StepOutcome.continue_with({"phase": "send"})
    outcome = outcome.model_copy(
        update={
            "requested_actions": [
                {"tool_id": "email.send", "idempotency_key": key, "args": {"to": "x"}},
            ],
        },
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code is None
    assert invoke_count == 1
    assert ledger.records()[0].status == SideEffectStatus.COMMITTED
    assert record.step_record is not None
    execution = record.step_record.diagnostics.get("declarative_tool_execution")
    assert execution is not None
    assert execution[0]["status"] == "success"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_replay_skips_declarative_invoke_on_resume() -> None:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-resume",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target="email.send",
        args={"to": "x"},
    )
    ledger.register(
        idempotency_key=key,
        run_id="run-resume",
        step_index=0,
        target="email.send",
    )
    ledger.commit(key, external_ref="msg-existing")
    invoke_count = 0

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal invoke_count
        invoke_count += 1
        return DeclarativeToolInvokeResult(status="success")

    step_ctx = AgentStepContext(
        step_index=0,
        side_effect_mode=SideEffectMode.DECLARATIVE,
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-resume",
        side_effect_mode=SideEffectMode.DECLARATIVE,
        policy_engine=PolicyEngine(),
        tool_profiles=build_profile_map([_MUTATING_TOOL]),
        side_effect_ledger=ledger,
        declarative_tool_invoker=CallableDeclarativeToolInvoker(_invoke),
    )
    outcome = StepOutcome.continue_with({"phase": "send"})
    outcome = outcome.model_copy(
        update={
            "requested_actions": [
                {"tool_id": "email.send", "idempotency_key": key, "args": {"to": "x"}},
            ],
        },
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code is None
    assert invoke_count == 0
    assert record.step_record is not None
    execution = record.step_record.diagnostics.get("declarative_tool_execution")
    assert execution is not None
    assert execution[0]["status"] == "replay_skipped"
    assert execution[0]["external_ref"] == "msg-existing"


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_compensates_after_policy_post_denies_committed_tools() -> None:
    ledger = SideEffectLedger()
    key = build_default_idempotency_key(
        run_id="run-comp",
        step_index=0,
        kind=SideEffectKind.TOOL,
        target="email.send",
        args={"to": "x"},
    )
    invoked: list[str] = []

    async def _invoke(**kwargs):  # type: ignore[no-untyped-def]
        invoked.append(kwargs["tool_id"])
        return DeclarativeToolInvokeResult(
            status="success",
            external_ref=f"ref-{kwargs['tool_id']}",
        )

    profiles = {
        "email.send": ToolExecutionProfile(
            tool_id="email.send",
            mutability=ToolMutability.MUTATING,
            reversibility=ToolReversibility.COMPENSATABLE,
            requires_idempotency_key=True,
            compensation_tool_id="email.recall",
        ),
    }
    step_ctx = AgentStepContext(
        step_index=0,
        side_effect_mode=SideEffectMode.DECLARATIVE,
    )
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        run_id="run-comp",
        side_effect_mode=SideEffectMode.DECLARATIVE,
        policy_engine=PolicyEngine(),
        tool_profiles=profiles,
        side_effect_ledger=ledger,
        declarative_tool_invoker=CallableDeclarativeToolInvoker(_invoke),
    )
    outcome = StepOutcome.complete("", terminal_reason=TerminalReason.GOAL_MET)
    outcome = outcome.model_copy(
        update={
            "requested_actions": [
                {"tool_id": "email.send", "idempotency_key": key, "args": {"to": "x"}},
            ],
        },
    )
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert invoked == ["email.send", "email.recall"]
    assert ledger.records()[0].status == SideEffectStatus.COMPENSATED
    assert record.step_record is not None
    compensation = record.step_record.diagnostics.get("compensation_enqueue")
    assert compensation is not None
    assert compensation[0]["status"] == "compensated"
    assert len(kernel_ctx.compensation_requests) == 1


@pytest.mark.unit
@pytest.mark.gate
async def test_kernel_policy_post_denies_empty_terminal_output() -> None:
    step_ctx = AgentStepContext(step_index=0)
    kernel_ctx = StepKernelContext(
        agent_id="demo",
        policy_engine=PolicyEngine(),
    )
    outcome = StepOutcome.complete("", terminal_reason=TerminalReason.GOAL_MET)
    record = await HarnessKernel.execute_step(outcome, step_ctx, kernel_ctx)
    assert record.error_code == AgentRunErrorCode.POLICY_DENIED
    assert record.policy_post is not None
    assert record.policy_post.action == PolicyAction.DENY
