# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, List

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools import insert_context_before_last_user
from intergrax.runtime.drop_in_knowledge_mode.tracing.trace_models import TraceLevel
from intergrax.runtime.drop_in_knowledge_mode.tracing.websearch.websearch_summary import WebsearchSummaryDiagV1


class WebsearchStep(RuntimeStep):
    """
    Execute websearch (if configured) and inject web context into the LLM prompt.

    Responsibilities:
      - run websearch agent/executor using state.messages_for_llm and tools context
      - build web context messages with websearch_prompt_builder
      - inject web context messages before the last user message
      - append compact web context text into state.tools_context_parts (for tools agent)
      - debug + trace
    """

    async def run(self, state: RuntimeState) -> None:
        state.used_websearch = False

        cfg = state.context.config
        enabled = bool(cfg.enable_websearch)

        configured = (
            state.context.websearch_executor is not None
            and state.context.websearch_prompt_builder is not None
        )

        # Summary defaults
        used_websearch = False
        results_count = 0
        context_blocks_count = 0
        no_evidence = False
        error_type = None
        error_message = None
        context_preview = ""
        context_preview_chars = 0

        if not enabled or not configured:
            state.trace_event(
                component="engine",
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

        try:
            web_results = await state.context.websearch_executor.search_async(
                query=state.request.message,
                top_k=cfg.max_docs_per_query,
                language=None,
                top_n_fetch=None,
            )

            results_count = len(web_results or [])
            if not web_results:
                # enabled+configured but no results
                state.trace_event(
                    component="engine",
                    step="websearch",
                    message="Web search executed (no results).",
                    level=TraceLevel.INFO,
                    payload=WebsearchSummaryDiagV1(
                        enabled=True,
                        configured=True,
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

            state.used_websearch = True
            used_websearch = True

            bundle = await state.context.websearch_prompt_builder.build_websearch_prompt(
                web_results=web_results,
                user_query=state.request.message,
                run_id=state.run_id,
            )

            context_messages = bundle.context_messages or []
            insert_context_before_last_user(state, context_messages)

            # Build preview (first block only)
            web_context_texts: list[str] = []
            for msg in context_messages:
                if msg.content:
                    web_context_texts.append(msg.content)

            context_blocks_count = len(web_context_texts)

            preview = "\n\n".join(web_context_texts[:1])
            PREVIEW_LIMIT = 300
            context_preview = preview[:PREVIEW_LIMIT]
            context_preview_chars = len(context_preview)

            # Guardrail signal: did websearch produce any grounded evidence?
            no_evidence = "No answer-relevant evidence extracted" in (context_preview or "")

        except Exception as exc:
            error_type = type(exc).__name__
            error_message = str(exc)

        # Trace web search step summary
        state.trace_event(
            component="engine",
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
