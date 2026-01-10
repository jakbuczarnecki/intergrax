# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List, Optional

from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.planning.runtime_step_handlers import RuntimeStep
from intergrax.runtime.nexus.runtime_steps.tools import insert_context_before_last_user
from intergrax.runtime.nexus.tracing.trace_models import TraceLevel
from intergrax.runtime.nexus.tracing.memory.user_longterm_memory_summary import UserLongtermMemorySummaryDiagV1


class UserLongtermMemoryStep(RuntimeStep):
    """
    Retrieve user long-term memory (LTM) and inject as context messages.

    - Uses state.user_longterm_memory_result cache if available.
    - Uses state.context.session_manager.search_user_longterm_memory(...) for retrieval.
    - Uses state.context.user_longterm_memory_prompt_builder.build_user_longterm_memory_prompt(hits)
      to create context messages.
    """

    async def run(self, state: RuntimeState) -> None:
        state.used_user_longterm_memory = False

        cfg = state.context.config
        enabled = bool(cfg.enable_user_longterm_memory)

        # Defaults for summary
        reason: Optional[str] = None
        hits_count = 0
        context_blocks_count = 0
        context_preview = ""
        context_preview_chars = 0
        top_k = int(cfg.max_longterm_entries_per_query)

        if not enabled:
            reason = "disabled"
            state.trace_event(
                component="engine",
                step="user_longterm_memory",
                message="User long-term memory disabled; skipping.",
                level=TraceLevel.INFO,
                payload=UserLongtermMemorySummaryDiagV1(
                    enabled=False,
                    used_user_longterm_memory=False,
                    reason=reason,
                    hits_count=0,
                    top_k=top_k,
                    context_blocks_count=0,
                    context_preview_chars=0,
                    context_preview="",
                ),
            )
            return

        built = state.user_longterm_memory_result

        if built is None:
            session = state.session
            assert session is not None, "Session must be set before user long-term memory step."

            query = (state.request.message or "").strip()
            if not query:
                reason = "empty_query"
                state.trace_event(
                    component="engine",
                    step="user_longterm_memory",
                    message="User long-term memory skipped (empty query).",
                    level=TraceLevel.INFO,
                    payload=UserLongtermMemorySummaryDiagV1(
                        enabled=True,
                        used_user_longterm_memory=False,
                        reason=reason,
                        hits_count=0,
                        top_k=top_k,
                        context_blocks_count=0,
                        context_preview_chars=0,
                        context_preview="",
                    ),
                )
                return

            built = await state.context.session_manager.search_user_longterm_memory(
                user_id=session.user_id,
                query=query,
                top_k=cfg.max_longterm_entries_per_query,
                score_threshold=cfg.longterm_score_threshold,
            )
            state.user_longterm_memory_result = built

        built = built or {}
        ltm_info = built.get("debug") or {}
        hits = built.get("hits") or []

        state.used_user_longterm_memory = bool(ltm_info.get("used", bool(hits)))
        hits_count = len(hits)

        if not state.used_user_longterm_memory:
            reason = (ltm_info.get("reason") or "no_hits_or_not_used") if enabled else "disabled"
            state.trace_event(
                component="engine",
                step="user_longterm_memory",
                message="User long-term memory executed but not used.",
                level=TraceLevel.INFO,
                payload=UserLongtermMemorySummaryDiagV1(
                    enabled=True,
                    used_user_longterm_memory=False,
                    reason=reason,
                    hits_count=hits_count,
                    top_k=top_k,
                    context_blocks_count=0,
                    context_preview_chars=0,
                    context_preview="",
                ),
            )
            return

        bundle = state.context.user_longterm_memory_prompt_builder.build_user_longterm_memory_prompt(hits)
        context_messages = bundle.context_messages or []
        insert_context_before_last_user(state, context_messages)

        # Build compact preview (first 1-2 content blocks)
        ltm_context_texts: list[str] = []
        for msg in context_messages:
            if msg.content:
                ltm_context_texts.append(msg.content)

        context_blocks_count = len(ltm_context_texts)

        preview = "\n\n".join(ltm_context_texts[:2])
        # Hard limit (keep it stable and safe)
        PREVIEW_LIMIT = 300
        context_preview = preview[:PREVIEW_LIMIT]
        context_preview_chars = len(context_preview)

        state.trace_event(
            component="engine",
            step="user_longterm_memory",
            message="User long-term memory step executed.",
            level=TraceLevel.INFO,
            payload=UserLongtermMemorySummaryDiagV1(
                enabled=True,
                used_user_longterm_memory=True,
                reason=None,
                hits_count=hits_count,
                top_k=top_k,
                context_blocks_count=context_blocks_count,
                context_preview_chars=context_preview_chars,
                context_preview=context_preview,
            ),
        )
