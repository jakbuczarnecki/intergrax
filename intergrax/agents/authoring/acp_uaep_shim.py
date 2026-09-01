# © Artur Czarnecki. All rights reserved.

"""UAEP run_step shim for migrated CognitiveAgent classes (ACP-MIG / Wave 8)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from intergrax.agents.authoring.state_merge import extract_acp_state_blob
from intergrax.contracts.acp_metadata_keys import AcpRunContextKey
from intergrax.contracts.acp_state import ACP_STATE_KEY
from intergrax.contracts.agent_decision import AgentDecision, AgentDecisionType
from intergrax.contracts.agent_run_enums import StepNextAction, TerminalReason
from intergrax.contracts.agent_step import AgentStep, StepOutput
from intergrax.contracts.agent_step_context import AgentStepContext
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.contracts.agent_run import AgentRunRequest
from intergrax.contracts.runtime_execution_context import RuntimeExecutionContext
from intergrax.runtime.kernel.step_kernel import StepKernelContext
from intergrax.runtime.nexus.engine.runtime_context import RuntimeContext
from intergrax.runtime.nexus.responses.response_schema import (
    RuntimeAnswer,
    RuntimeRequest,
    RuntimeStats,
)

if TYPE_CHECKING:
    from intergrax.agents.authoring.patterns.base import CognitiveAgent


def apply_host_tool_invoker_to_runtime_context(
    runtime_context: RuntimeContext,
    request_metadata: dict[str, Any],
) -> None:
    """Overlay Tier-3 host catalog wiring onto agent stub ``RuntimeContext``."""
    from intergrax.agents.persistence.catalog_declarative_invoker import CatalogDeclarativeToolInvoker
    from intergrax.agents.persistence.tool_invoker_wiring import (
        resolve_declarative_tool_invoker_from_metadata,
    )
    from intergrax.runtime.nexus.tools.catalog_dispatch import resolve_tool_registry

    invoker = resolve_declarative_tool_invoker_from_metadata(request_metadata)
    if not isinstance(invoker, CatalogDeclarativeToolInvoker):
        return
    tool_invoker = invoker.tool_invoker
    registry = resolve_tool_registry(tool_invoker)
    if registry is None:
        return
    runtime_context.config.tool_registry = registry
    runtime_context.config.tool_invoker = tool_invoker

    from intergrax.applications._shared.rag_runtime_bridge import apply_rag_from_tool_wiring_context
    from intergrax.tools.core.handler import WiringContextToolHandler

    wiring_ctx = None
    for registered in registry.list():
        handler = registered.handler
        if isinstance(handler, WiringContextToolHandler):
            wiring_ctx = handler._ctx
            break
    if wiring_ctx is None:
        return
    runtime_context.config.tool_wiring_context = wiring_ctx
    runtime_context.config.enable_rag = True
    if wiring_ctx.integration_profile is not None:
        runtime_context.config.integration_profile = wiring_ctx.integration_profile
    apply_rag_from_tool_wiring_context(runtime_context.config, wiring_ctx)


def _attach_functional_evidence_recorder_from_runtime_state(
    exec_ctx: RuntimeExecutionContext,
) -> None:
    from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
        attach_functional_evidence_recorder_from_runtime_state,
    )

    attach_functional_evidence_recorder_from_runtime_state(exec_ctx)


def attach_acp_catalog_exec_ctx(
    step_ctx: AgentStepContext,
    *,
    kernel_ctx: StepKernelContext,
    request: AgentRunRequest,
    contract: AgentContract,
) -> None:
    """Bridge ACP session steps to ``uaep_exec_ctx`` for catalog tool invocation."""
    if isinstance(step_ctx.metadata.get("uaep_exec_ctx"), RuntimeExecutionContext):
        return

    from intergrax.agents.authoring.acp_stub_reflex import build_agent_runtime_context
    from intergrax.agents.authoring.stub_llm import PrefixStubLLMAdapter
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
    from intergrax.runtime.nexus.tools.uaep_tool_gateway import BoundToolGateway

    from intergrax.contracts.execution_identity import (
        require_active_execution_id,
        require_active_execution_identity,
        validate_run_id,
        validate_task_id,
    )

    resolved_run_id = validate_run_id(step_ctx.run_id)
    resolved_task_id = validate_task_id(step_ctx.task_id)
    run_id, attempt_id = require_active_execution_identity()
    if run_id != resolved_run_id:
        raise RuntimeError("step run_id conflicts with active execution identity")
    execution_id = require_active_execution_id()
    runtime_request = RuntimeRequest(
        agent_id=contract.id,
        tenant_id=str(request.identity.tenant_id or step_ctx.tenant_id or "default"),
        user_id=str(request.identity.user_id or ""),
        session_id=str(request.session_id or resolved_run_id),
        message=str(request.input or step_ctx.message or ""),
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        metadata=dict(request.metadata),
    )
    runtime_request.metadata.setdefault("run_id", resolved_run_id)
    runtime_request.metadata.setdefault("task_id", resolved_task_id)

    exec_ctx = RuntimeExecutionContext(
        task_id=resolved_task_id,
        run_id=resolved_run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
        agent_id=contract.id,
        contract=contract,
        request=runtime_request,
    )
    runtime_context = build_agent_runtime_context(
        runtime_request,
        PrefixStubLLMAdapter(prefix=contract.id),
    )
    apply_host_tool_invoker_to_runtime_context(runtime_context, request.metadata)
    exec_ctx.metadata["runtime_state"] = RuntimeState(
        context=runtime_context,
        request=runtime_request,
        run_id=resolved_run_id,
    )
    allowed_tools_raw = step_ctx.metadata.get("allowed_tools")
    if isinstance(allowed_tools_raw, list) and allowed_tools_raw:
        allowed_tools = [str(tool_id) for tool_id in allowed_tools_raw if str(tool_id).strip()]
    else:
        allowed_tools = list(contract.allowed_tools)
    if kernel_ctx.declarative_tool_invoker is not None:
        exec_ctx.tool_gateway = BoundToolGateway(
            exec_ctx,
            allowed_tools=allowed_tools,
        )
    from intergrax.runtime.workspace.exec_ctx_isolation import attach_isolation_to_exec_ctx

    attach_isolation_to_exec_ctx(exec_ctx, runtime_request, task_id=resolved_task_id)
    from intergrax.runtime.observability.functional_evidence_runtime_wiring import (
        attach_functional_evidence_recorder_from_runtime_state,
    )

    attach_functional_evidence_recorder_from_runtime_state(exec_ctx)
    step_ctx.metadata["uaep_exec_ctx"] = exec_ctx


def close_acp_catalog_exec_ctx(step_ctx: AgentStepContext) -> None:
    """Release per-step ACP catalog ``RuntimeContext`` (tool execution pool)."""
    raw = step_ctx.metadata.pop("uaep_exec_ctx", None)
    if not isinstance(raw, RuntimeExecutionContext):
        return
    runtime_state = raw.metadata.get("runtime_state")
    from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

    if isinstance(runtime_state, RuntimeState):
        runtime_state.context.close()


def _run_input_from_request(exec_ctx: RuntimeExecutionContext) -> str | dict[str, Any]:
    request = exec_ctx.request
    if request is None:
        return ""
    if isinstance(request, RuntimeRequest):
        message = request.message
        if message.strip():
            return message
        metadata = request.metadata
        if metadata.get("message"):
            return str(metadata["message"])
        return message or ""
    metadata = request.metadata
    if metadata.get("message"):
        return str(metadata["message"])
    return ""


def build_step_context_from_uaep(
    agent: CognitiveAgent,
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
) -> AgentStepContext:
    state_root: dict[str, Any] = {}
    raw_state = exec_ctx.metadata.get(ACP_STATE_KEY)
    if isinstance(raw_state, dict):
        state_root = {ACP_STATE_KEY: dict(raw_state)}
    return AgentStepContext(
        step_index=step.step_index,
        run_id=exec_ctx.run_id,
        agent_id=exec_ctx.agent_id,
        contract_id=exec_ctx.contract.id if exec_ctx.contract else exec_ctx.agent_id,
        state_snapshot=state_root,
        metadata={
            AcpRunContextKey.RUN_INPUT: _run_input_from_request(exec_ctx),
            "uaep_exec_ctx": exec_ctx,
            "task_id": exec_ctx.task_id,
        },
    )


def step_output_from_outcome(step: AgentStep, outcome: Any) -> StepOutput:
    if isinstance(outcome.output, dict):
        summary = str(outcome.output.get("summary") or outcome.output.get("answer") or "")
        data = {key: value for key, value in outcome.output.items() if key != "summary"}
    else:
        summary = str(outcome.output or "")
        data = {}
    return StepOutput(step_id=step.step_id, summary=summary, data=data)


def agent_decision_from_outcome(outcome: Any) -> AgentDecision:
    if outcome.next_action == StepNextAction.PAUSE_HITL:
        return AgentDecision(
            type=AgentDecisionType.REQUEST_HUMAN,
            reason=outcome.diagnostics.get("pause_reason", "human_required")
            if outcome.diagnostics
            else "human_required",
        )
    if outcome.next_action == StepNextAction.FAIL:
        return AgentDecision(
            type=AgentDecisionType.FAIL,
            reason=outcome.terminal_reason.value if outcome.terminal_reason else "failed",
        )
    if outcome.next_action == StepNextAction.REPLAN:
        return AgentDecision(
            type=AgentDecisionType.MODIFY_PLAN,
            reason=outcome.terminal_reason.value if outcome.terminal_reason else "replan",
        )
    if outcome.is_terminal or outcome.next_action == StepNextAction.CONTINUE:
        if outcome.is_terminal:
            return AgentDecision(
                type=AgentDecisionType.COMPLETE,
                reason=outcome.terminal_reason.value if outcome.terminal_reason else TerminalReason.GOAL_MET.value,
            )
    return AgentDecision(type=AgentDecisionType.CONTINUE, reason="continue")


async def execute_cognitive_step_via_acp(
    agent: CognitiveAgent,
    step: AgentStep,
    exec_ctx: RuntimeExecutionContext,
) -> StepOutput:
    step_ctx = build_step_context_from_uaep(agent, step, exec_ctx)
    outcome = await agent.on_next_step(step_ctx)
    exec_ctx.metadata[AcpRunContextKey.LAST_OUTCOME] = outcome.model_dump(mode="json")
    acp_blob = extract_acp_state_blob(step_ctx.state_snapshot)
    exec_ctx.metadata[ACP_STATE_KEY] = acp_blob
    step_output = step_output_from_outcome(step, outcome)
    exec_ctx.metadata["runtime_answer"] = RuntimeAnswer(
        run_id=exec_ctx.run_id,
        answer=step_output.summary,
        stats=RuntimeStats(total_tokens=0, duration_ms=0, extra={"cost": 0.0}),
    )
    return step_output


def decide_after_cognitive_step(
    exec_ctx: RuntimeExecutionContext,
    *,
    default_reason: str = "cognitive step finished",
) -> AgentDecision:
    raw = exec_ctx.metadata.get(AcpRunContextKey.LAST_OUTCOME)
    if raw is None:
        return AgentDecision(type=AgentDecisionType.COMPLETE, reason=default_reason)
    from intergrax.agents.authoring.step_outcome import StepOutcome

    outcome = StepOutcome.model_validate(raw)
    return agent_decision_from_outcome(outcome)
