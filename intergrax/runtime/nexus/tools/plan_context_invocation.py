# © Artur Czarnecki. All rights reserved.

"""RAG, websearch, and planner-tool invocation for ToolRuntime (replaces legacy pipeline steps)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Optional

from intergrax.llm.messages import ChatMessage
from intergrax.prompts.registry.prompt_registry_resolver import resolve_yaml_prompt_registry
from intergrax.runtime.nexus.budget.budget_ticks import (
    record_rag_invocation_and_enforce,
    record_websearch_invocation_and_enforce,
)
from intergrax.runtime.nexus.context.tool_context_helpers import (
    format_rag_context,
    insert_context_before_last_user,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    merge_provider_metadata_into_request,
)
from intergrax.runtime.nexus.tools.catalog_context import (
    build_rag_retrieve_input,
    build_websearch_query_input,
    invoke_catalog_context_tool,
)
from intergrax.runtime.nexus.tools.catalog_dispatch import resolve_tool_registry
from intergrax.runtime.nexus.tools.tool_loop import (
    inject_tool_traces_system_context,
    run_bounded_tool_loop,
)
from intergrax.runtime.nexus.tools.tool_planner_input import resolve_tool_planner_input
from intergrax.runtime.nexus.tools.adaptive_tool_mode_resolver import recommend_tool_modes
from intergrax.runtime.nexus.tools.tool_selection import (
    SemanticToolIndexSelectionStrategy,
    ToolSelectionContext,
    resolve_planner_allowed_tool_ids_async,
    resolve_selection_strategy,
    strategy_trace_id,
)
from intergrax.runtime.nexus.tracing.rag.rag_summary import RagSummaryDiagV1
from intergrax.runtime.nexus.tracing.tools.tool_selection import (
    ToolSelectionCandidateDiagV1,
    ToolSelectionDiagV1,
)
from intergrax.runtime.nexus.tracing.tools.tools_summary import ToolsSummaryDiagV1
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel
from intergrax.runtime.nexus.tracing.websearch.websearch_summary import WebsearchSummaryDiagV1
from intergrax.tools.unified.constants import RAG_RETRIEVE_TOOL_ID, WEBSEARCH_QUERY_TOOL_ID
from intergrax.websearch.schemas.web_search_result import WebSearchResult


def _selection_candidates(
    strategy: object,
    candidate_ids: Sequence[str],
) -> list[ToolSelectionCandidateDiagV1]:
    if isinstance(strategy, SemanticToolIndexSelectionStrategy) and strategy.last_ranks:
        return [
            ToolSelectionCandidateDiagV1(tool_id=tool_id, score=score)
            for tool_id, score in strategy.last_ranks
        ]
    return [ToolSelectionCandidateDiagV1(tool_id=tool_id) for tool_id in candidate_ids]


async def run_rag_context(state: RuntimeState) -> None:
    state.used_rag = False
    ctx = state.context

    if not ctx.config.enable_rag:
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="rag",
            message="RAG disabled; skipping.",
            level=TraceLevel.INFO,
            payload=RagSummaryDiagV1(
                rag_enabled=False,
                used_rag=False,
                chunks_count=0,
                context_messages_count=0,
                warning=None,
            ),
        )
        return

    if ctx.context_builder is None:
        raise RuntimeError("RAG enabled but ContextBuilder is not configured.")

    record_rag_invocation_and_enforce(state)

    if invoke_catalog_context_tool(
        state,
        RAG_RETRIEVE_TOOL_ID,
        build_rag_retrieve_input(state),
        step_id="rag/catalog",
    ):
        if state.used_rag:
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="rag",
                message="RAG context built via catalog tool rag.retrieve.",
                level=TraceLevel.INFO,
                payload=RagSummaryDiagV1(
                    rag_enabled=True,
                    used_rag=True,
                    chunks_count=0,
                    context_messages_count=0,
                    warning=None,
                ),
            )
        return

    built = state.context_builder_result
    if built is None:
        session = state.session
        assert session is not None, "Session must be set before RAG invocation."
        built = await ctx.context_builder.build_context(
            session=session,
            request=state.request,
            base_history=state.base_history,
        )
        state.context_builder_result = built

    retrieved_chunks = built.retrieved_chunks or []
    security_profile = ctx.config.security_profile
    if security_profile is not None and security_profile.retrieval_poisoning_defense_enabled:
        from intergrax.runtime.architecture.retrieval_security_wiring import (
            filter_retrieved_chunks_for_poisoning,
        )

        filtered_chunks, review_warnings = filter_retrieved_chunks_for_poisoning(retrieved_chunks)
        if len(filtered_chunks) != len(retrieved_chunks):
            retrieved_chunks = filtered_chunks
            built = replace(
                built,
                retrieved_chunks=retrieved_chunks,
                rag_used=bool(retrieved_chunks),
                rag_reason=built.rag_reason or "retrieval_poisoning_quarantine",
            )
            state.context_builder_result = built

    state.used_rag = built.rag_used
    rag_reason = built.rag_reason

    if not state.used_rag:
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="rag",
            message="RAG enabled but not used (no retrieved context).",
            level=TraceLevel.INFO,
            payload=RagSummaryDiagV1(
                rag_enabled=True,
                used_rag=False,
                chunks_count=0,
                context_messages_count=0,
                warning=rag_reason,
            ),
        )
        return

    if ctx.rag_prompt_builder is None:
        raise RuntimeError("RAG enabled but rag_prompt_builder is not configured.")

    bundle = ctx.rag_prompt_builder.build_rag_prompt(built)
    context_messages = bundle.context_messages or []
    if context_messages:
        insert_context_before_last_user(state, context_messages)

    rag_context_text = format_rag_context(retrieved_chunks)
    if rag_context_text:
        state.tools_context_parts.append("RAG CONTEXT:\n" + rag_context_text)

    state.trace_event(
        component=TraceComponent.ENGINE,
        step="rag",
        message="RAG context built and injected.",
        level=TraceLevel.INFO,
        payload=RagSummaryDiagV1(
            rag_enabled=True,
            used_rag=True,
            chunks_count=len(retrieved_chunks),
            context_messages_count=len(context_messages),
            warning=None,
        ),
    )
    merge_provider_metadata_into_request(state)


async def run_websearch_context(state: RuntimeState) -> None:
    state.used_websearch = False
    cfg = state.context.config
    enabled = bool(cfg.enable_websearch)
    configured = (
        state.context.websearch_executor is not None
        and state.context.websearch_prompt_builder is not None
    )

    used_websearch = False
    results_count = 0
    context_blocks_count = 0
    no_evidence = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    context_preview = ""
    context_preview_chars = 0
    preview_limit = 300
    error_limit = 500

    if not enabled or not configured:
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="websearch",
            message="Web search skipped (disabled or not configured).",
            level=TraceLevel.INFO,
            payload=WebsearchSummaryDiagV1(
                enabled=enabled,
                configured=configured,
                used_websearch=False,
                results_count=0,
                context_blocks_count=0,
                no_evidence=False,
                error_type=None,
                error_message=None,
                context_preview_chars=0,
                context_preview="",
            ),
        )
        return

    record_websearch_invocation_and_enforce(state)

    if invoke_catalog_context_tool(
        state,
        WEBSEARCH_QUERY_TOOL_ID,
        build_websearch_query_input(state),
        step_id="websearch/catalog",
    ):
        state.trace_event(
            component=TraceComponent.ENGINE,
            step="websearch",
            message="Web search executed via catalog tool websearch.query.",
            level=TraceLevel.INFO,
            payload=WebsearchSummaryDiagV1(
                enabled=True,
                configured=True,
                used_websearch=bool(state.used_websearch),
                results_count=0,
                context_blocks_count=1 if state.used_websearch else 0,
                no_evidence=not state.used_websearch,
                error_type=None,
                error_message=None,
                context_preview_chars=0,
                context_preview="",
            ),
        )
        return

    web_results: list[WebSearchResult] = []
    context_messages: list[ChatMessage] = []

    try:
        web_results = await state.context.websearch_executor.search_async(
            query=state.request.message,
            top_k=cfg.max_docs_per_query,
            language=None,
            top_n_fetch=None,
        )
        results_count = len(web_results or [])
        if not web_results:
            no_evidence = True
            state.trace_event(
                component=TraceComponent.ENGINE,
                step="websearch",
                message="Web search executed (no results).",
                level=TraceLevel.INFO,
                payload=WebsearchSummaryDiagV1(
                    enabled=True,
                    configured=True,
                    used_websearch=False,
                    results_count=0,
                    context_blocks_count=0,
                    no_evidence=True,
                    error_type=None,
                    error_message=None,
                    context_preview_chars=0,
                    context_preview="",
                ),
            )
            return

        bundle = await state.context.websearch_prompt_builder.build_websearch_prompt(
            web_results=web_results,
            user_query=state.request.message,
            run_id=state.run_id,
        )
        no_evidence = bool(bundle.no_evidence) or (bundle.sources_count == 0)
        context_messages = bundle.context_messages or []
        if context_messages:
            insert_context_before_last_user(state, context_messages)
            state.used_websearch = True
            used_websearch = True

        web_context_texts: list[str] = []
        for msg in context_messages:
            if msg.content:
                web_context_texts.append(msg.content)
        context_blocks_count = len(web_context_texts)
        if context_blocks_count == 0:
            no_evidence = True
        if web_context_texts:
            preview = web_context_texts[0]
            context_preview = (preview or "")[:preview_limit]
            context_preview_chars = len(context_preview)
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)[:error_limit]

    state.trace_event(
        component=TraceComponent.ENGINE,
        step="websearch",
        message="Web search step executed.",
        level=TraceLevel.ERROR if error_type else TraceLevel.INFO,
        payload=WebsearchSummaryDiagV1(
            enabled=enabled,
            configured=configured,
            used_websearch=used_websearch if error_type is None else False,
            results_count=results_count,
            context_blocks_count=context_blocks_count,
            no_evidence=no_evidence,
            error_type=error_type,
            error_message=error_message,
            context_preview_chars=context_preview_chars,
            context_preview=context_preview,
        ),
    )
    merge_provider_metadata_into_request(state)


async def run_tools_context(state: RuntimeState) -> None:
    state.used_tools = False
    state.tool_traces = []
    state.tool_planner_answer = None

    invoker = state.context.config.tool_invoker
    tool_planner = state.context.config.tool_planner
    tools_mode = state.context.config.tools_mode

    if invoker is None or tool_planner is None or tools_mode == "off":
        return

    warning: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    loop_pattern_id: Optional[str] = None
    loop_stop_reason: Optional[str] = None
    tool_selection_mode = state.context.config.tool_selection_mode
    tool_invocation_mode = state.context.config.tool_invocation_mode

    try:
        planner_input = resolve_tool_planner_input(state)
        registry = resolve_tool_registry(invoker)
        if registry is not None:
            hook = state.context.config.tool_engine_hook
            if hook is not None and hook.enabled:
                recommendation = recommend_tool_modes(
                    registry=registry,
                    query=state.request.message or "",
                )
                tool_selection_mode = recommendation.tool_selection_mode
                if recommendation.tool_invocation_mode is not None:
                    tool_invocation_mode = recommendation.tool_invocation_mode
            selection_ctx = ToolSelectionContext(
                registry=registry,
                query=state.request.message or "",
                skill_profile=state.context.config.skill_profile,
                plan_allowed_tool_ids=state.tool_planner_allowed_tool_ids,
                top_k=state.context.config.tool_selection_top_k,
                max_hierarchy_passes=state.context.config.tool_selection_max_hierarchy_passes,
                embedding_manager=state.context.config.embedding_manager,
                hierarchical_llm_category_pass=state.context.config.tool_selection_hierarchical_llm_pass,
                llm_adapter=state.context.config.llm_adapter,
            )
            selection_strategy = resolve_selection_strategy(
                tool_selection_mode,
                selection_ctx,
                strategy_override=state.context.config.tool_selection_strategy,
                entry_point_strategy_id=state.context.config.tool_selection_strategy_id,
            )
            allowed_tool_ids = await resolve_planner_allowed_tool_ids_async(
                tool_selection_mode,
                selection_ctx,
                strategy_override=selection_strategy,
            )
            candidate_ids = tuple(allowed_tool_ids or ())
            candidates = list(_selection_candidates(selection_strategy, candidate_ids))
            state.trace_event(
                component=TraceComponent.TOOLS,
                step="tool_selection",
                message="Tool selection strategy resolved planner allow-list.",
                level=TraceLevel.INFO,
                payload=ToolSelectionDiagV1(
                    strategy_id=strategy_trace_id(selection_strategy),
                    selection_mode=tool_selection_mode.value,
                    candidate_tool_ids=list(candidate_ids),
                    candidates=candidates,
                ),
            )
        else:
            allowed_tool_ids = state.tool_planner_allowed_tool_ids

        loop_result = run_bounded_tool_loop(
            state=state,
            invoker=invoker,
            tool_planner=tool_planner,
            planner_input=planner_input,
            allowed_tool_ids=allowed_tool_ids,
            max_iterations=state.context.config.max_tool_iterations,
            invocation_mode=tool_invocation_mode,
        )
        loop_pattern_id = loop_result.pattern_id or None
        loop_stop_reason = loop_result.stop_reason or None

        if not loop_result.tool_traces:
            if tools_mode == "required":
                from intergrax.runtime.nexus.errors.tools_required_error import ToolsRequiredError

                raise ToolsRequiredError(run_id=state.run_id)
        else:
            state.used_tools = True
            state.tool_traces = list(loop_result.tool_traces)

        if loop_result.used_native_tool_messages and loop_result.appended_messages:
            state.messages_for_llm.extend(loop_result.appended_messages)
        elif state.tool_traces:
            registry = resolve_yaml_prompt_registry(
                registry=state.context.prompt_registry,
                catalog_path=state.context.config.prompt_catalog_path,
            )
            localized = registry.resolve_localized("tools_runtime_context")
            inject_tool_traces_system_context(
                state,
                state.tool_traces,
                runtime_context_prompt=localized.system,
                aggregate=loop_result.aggregate,
            )
    except Exception as exc:
        error_type = type(exc).__name__
        error_message = str(exc)

    tool_names = sorted({t.tool_name for t in state.tool_traces if t.tool_name})
    state.trace_event(
        component=TraceComponent.ENGINE,
        step="tools",
        message="Tools planner + runtime execution executed.",
        level=TraceLevel.ERROR if error_type else TraceLevel.INFO,
        payload=ToolsSummaryDiagV1(
            tools_mode=tools_mode,
            used_tools=state.used_tools,
            tool_calls_count=len(state.tool_traces),
            tool_names=tool_names,
            warning=warning,
            error_type=error_type,
            error_message=error_message,
            pattern_id=loop_pattern_id,
            stop_reason=loop_stop_reason,
        ),
    )
    merge_provider_metadata_into_request(state)
