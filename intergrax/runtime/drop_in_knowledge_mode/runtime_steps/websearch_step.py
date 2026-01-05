# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Any, List

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.runtime_steps.tools import insert_context_before_last_user


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
        state.websearch_debug = {}
        state.used_websearch = False

        if (
            not state.context.config.enable_websearch
            or state.context.websearch_executor is None
            or state.context.websearch_prompt_builder is None
        ):
            return

        try:
            web_results = await state.context.websearch_executor.search_async(
                query=state.request.message,
                top_k=state.context.config.max_docs_per_query,
                language=None,
                top_n_fetch=None,
            )

            state.set_debug_section("websearch", {})
            dbg = state.debug_trace["websearch"]

            raw_preview = []
            for d in (web_results or [])[:5]:
                raw_preview.append(
                    {
                        "type": type(d).__name__,
                        "title": d.title,
                        "url": d.url,
                        "snippet_len": len(d.snippet or ""),
                        "text_len": len(d.text or ""),
                    }
                )

            dbg["raw_results_preview"] = raw_preview

            if not web_results:
                return

            state.used_websearch = True

            bundle = await state.context.websearch_prompt_builder.build_websearch_prompt(
                web_results=web_results,
                user_query=state.request.message,
                run_id=state.run_id,
            )

            context_messages = bundle.context_messages or []
            insert_context_before_last_user(state, context_messages)
            state.websearch_debug.update(bundle.debug_info or {})

            # Debug trace (no tools coupling)
            web_context_texts: List[str] = []
            for msg in context_messages:
                if msg.content:
                    web_context_texts.append(msg.content)

            dbg["context_blocks_count"] = len(web_context_texts)

            # Preview only to avoid bloating trace
            preview = "\n\n".join(web_context_texts[:1])
            dbg["context_preview"] = preview
            dbg["context_preview_chars"] = len(preview)

            # Guardrail signal: did websearch produce any grounded evidence?
            dbg["no_evidence"] = "No answer-relevant evidence extracted" in (preview or "")
            state.websearch_debug["no_evidence"] = dbg["no_evidence"]

            # Optional: doc-level preview (titles/urls only)
            dbg["docs_preview"] = [
                {
                    "title": d.title,
                    "url": d.url,
                }
                for d in (web_results or [])[:5]
            ]

        except Exception as exc:
            state.websearch_debug["error"] = str(exc)

        # Trace web search step.
        state.trace_event(
            component="engine",
            step="websearch",
            message="Web search step executed.",
            data={
                "websearch_enabled": state.context.config.enable_websearch,
                "used_websearch": state.used_websearch,
                "has_error": "error" in (state.websearch_debug or {}),
                "no_evidence": state.websearch_debug.get("no_evidence", False),
            },
        )