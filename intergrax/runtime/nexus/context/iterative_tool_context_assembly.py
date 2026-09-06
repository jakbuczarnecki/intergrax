# © Artur Czarnecki. All rights reserved.

"""Canonical CE assembly for iterative native tool-feedback rounds (UE-6C)."""

from __future__ import annotations

from collections.abc import Sequence

from intergrax.contracts.model_visible_evidence import ModelVisibleEvidenceReference
from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextBudgetSnapshot,
    ContextDecisionSnapshot,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.protocols import ContextEngine
from intergrax.contracts.context_assembly import TaskContextAssemblyOptions
from intergrax.context.providers.legacy_bridge import TOOL_OUTPUT_BLOCKS_HANDLE
from intergrax.llm.messages import ChatMessage
from intergrax.runtime.nexus.budget.budget_ticks import (
    enforce_wall_time_budget,
    record_planner_iteration_and_enforce,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.tools.investigation_proof import (
    InvestigationProof,
    InvestigationProofStep,
    build_investigation_proof_step,
    collect_available_evidence_ids,
    prepare_native_planner_messages_with_follow_up_context,
)
from intergrax.runtime.nexus.tools.invoker import RuntimeToolInvoker
from intergrax.runtime.nexus.tools.native_tool_plan_alignment import (
    validate_native_tool_plan_alignment,
)
from intergrax.runtime.nexus.tools.tool_invocation_pattern import (
    ToolInvocationResult,
    ToolInvocationStopReason,
)
from intergrax.runtime.nexus.tools.tool_loop import (
    _coerce_messages,
    append_assistant_tool_call_message,
    execute_planned_tool_calls,
    record_identical_tool_call_fingerprints,
    tool_output_blocks_from_native_round,
    validate_identical_tool_call_repeats,
)
from intergrax.runtime.nexus.tools.tool_planning_policy import tool_choice_for_mode
from intergrax.runtime.nexus.tools.tool_planner_protocol import IterativeToolPlannerProtocol


async def assemble_iterative_tool_planner_messages(
    state: RuntimeState,
    engine: ContextEngine,
    messages: list[ChatMessage],
) -> tuple[ChatMessage, ...]:
    """Run canonical CE assembly for the next bounded ReAct planner round."""
    assembly_request = ContextAssemblyRequest(
        trace_id=state.run_id,
        run_id=state.run_id,
        task_id=state.task_id,
        tenant_id=state.tenant_id,
        assembly_scope="acp_step",
        objective=state.request.message or "",
        decision_profile=ContextDecisionSnapshot(),
        budget_policy=ContextBudgetSnapshot(max_chars=16_000, max_tokens_estimate=4_000),
        assembly_options=TaskContextAssemblyOptions(),
        step_kind="tool_call",
        excluded_sources=frozenset({ContextFragmentSource.SESSION_HISTORY}),
    )
    runtime_config = state.context.config
    handles: dict[str, object] = {
        "runtime_config": runtime_config,
        "messages": list(messages),
        "event_bus": runtime_config.runtime_event_bus,
        "node_id": state.request.metadata.get("graph_node_id") or state.request.agent_id,
        "agent_id": state.request.agent_id,
        "engine_id": engine.engine_id,
        TOOL_OUTPUT_BLOCKS_HANDLE: list(state.iterative_tool_output_blocks),
    }
    provider_ctx = ContextProviderContext(engine_id=engine.engine_id, handles=handles)
    assembled = await engine.assemble(assembly_request, provider_ctx=provider_ctx)
    return assembled.messages


async def run_ce_bounded_tool_loop(
    *,
    state: RuntimeState,
    invoker: RuntimeToolInvoker,
    tool_planner: IterativeToolPlannerProtocol,
    planner_input: str | list[ChatMessage],
    allowed_tool_ids: Sequence[str] | None,
    max_iterations: int,
    prior_model_visible_references: Sequence[ModelVisibleEvidenceReference] = (),
) -> ToolInvocationResult:
    """Bounded ReAct with tool feedback routed through Context Engineering."""
    max_iters = max(1, int(max_iterations))
    engine = state.context.config.context_engine
    if engine is None:
        raise RuntimeError("context_engine is required for CE bounded tool loop")

    messages = _coerce_messages(planner_input)
    initial_len = len(messages)
    appended: list[ChatMessage] = []
    all_traces: list = []
    iterations = 0
    stop_reason: ToolInvocationStopReason = "max_iterations"
    fingerprint_counts: dict[str, int] = {}
    loop_cfg = state.context.config
    proof_steps: list[InvestigationProofStep] = []

    while iterations < max_iters:
        if iterations >= 1:
            enforce_wall_time_budget(state)
        record_planner_iteration_and_enforce(state)
        iterations += 1

        planner_messages = list(
            await assemble_iterative_tool_planner_messages(state, engine, messages)
        )
        planning_messages = prepare_native_planner_messages_with_follow_up_context(
            planner_messages,
            round_index=iterations,
            prior_model_visible_references=prior_model_visible_references,
        )
        llm_result, tool_plan = tool_planner.plan_native_round(
            planning_messages,
            allowed_tool_ids=allowed_tool_ids,
            run_id=state.run_id,
            tool_choice=tool_choice_for_mode(state.context.config.tools_mode),
        )

        if llm_result.content and not tool_plan.calls:
            stop_reason = "planner_final_answer"
            break

        if not tool_plan.calls:
            stop_reason = "empty_tool_calls"
            break

        validate_native_tool_plan_alignment(llm_result.tool_calls, tool_plan)

        proof_steps.append(
            build_investigation_proof_step(
                round_index=iterations,
                assistant_content=llm_result.content,
                tool_calls=llm_result.tool_calls,
                messages_before_round=planner_messages,
                prior_model_visible_references=prior_model_visible_references,
            )
        )

        validate_identical_tool_call_repeats(
            tool_plan.calls,
            fingerprint_counts=fingerprint_counts,
            max_repeats=loop_cfg.max_identical_tool_call_repeats,
        )
        record_identical_tool_call_fingerprints(tool_plan.calls, fingerprint_counts)
        round_outcomes = execute_planned_tool_calls(
            state=state,
            invoker=invoker,
            calls=tool_plan.calls,
            idempotency_prefix=f"{state.run_id}:loop{iterations}",
        )
        all_traces.extend(outcome.trace for outcome in round_outcomes)

        append_assistant_tool_call_message(
            messages,
            assistant_content=llm_result.content,
            tool_calls=llm_result.tool_calls,
        )
        state.iterative_tool_output_blocks.extend(
            tool_output_blocks_from_native_round(
                llm_result.tool_calls,
                tool_plan.calls,
                round_outcomes,
            )
        )

    investigation_proof: InvestigationProof | None = None
    if proof_steps:
        final_available_evidence_ids: tuple[str, ...] = ()
        if stop_reason == "planner_final_answer":
            evidence_messages = list(
                await assemble_iterative_tool_planner_messages(state, engine, messages)
            )
            final_available_evidence_ids = collect_available_evidence_ids(
                evidence_messages,
                prior_model_visible_references,
            )
        investigation_proof = InvestigationProof(
            steps=tuple(proof_steps),
            final_available_evidence_ids=final_available_evidence_ids,
        )

    final_assembled = await assemble_iterative_tool_planner_messages(state, engine, messages)
    appended = list(final_assembled[initial_len:])

    return ToolInvocationResult(
        tool_traces=all_traces,
        loop_iterations=iterations,
        stop_reason=stop_reason,
        appended_messages=appended,
        used_native_tool_messages=True,
        used_ce_tool_feedback=True,
        investigation_proof=investigation_proof,
        pattern_id="bounded_react",
    )
