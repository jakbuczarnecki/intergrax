# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_state import RuntimeState
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.contract import RuntimeStep
from intergrax.runtime.drop_in_knowledge_mode.engine.runtime_steps.tools import (
    insert_context_before_last_user,
)


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
        state.set_debug_value("user_longterm_memory_hits", 0)

        # Create a single debug section early and only mutate it later.
        state.set_debug_section("user_longterm_memory", {
            "enabled": bool(state.context.config.enable_user_longterm_memory),
            "used": False,
            "reason": None,
            "hits": 0,
            "retrieval_debug": {},
            "context_blocks_count": 0,
            "context_preview": "",
            "context_preview_chars": 0,
            "hits_preview": [],
        })
        dbg = state.debug_trace["user_longterm_memory"]

        if not state.context.config.enable_user_longterm_memory:
            dbg["reason"] = "disabled"
            return

        built = state.user_longterm_memory_result

        if built is None:
            session = state.session
            assert session is not None, "Session must be set before user long-term memory step."

            query = (state.request.message or "").strip()
            if not query:
                dbg["reason"] = "empty_query"
                dbg["used"] = False
                dbg["hits"] = 0
                state.set_debug_value("user_longterm_memory_hits", 0)
                return

            built = await state.context.session_manager.search_user_longterm_memory(
                user_id=session.user_id,
                query=query,
                top_k=state.context.config.max_longterm_entries_per_query,
                score_threshold=state.context.config.longterm_score_threshold,
            )
            state.user_longterm_memory_result = built

        built = built or {}
        ltm_info = built.get("debug") or {}
        hits = built.get("hits") or []

        # Keep retrieval debug inside the same section
        dbg["retrieval_debug"] = ltm_info

        state.used_user_longterm_memory = bool(ltm_info.get("used", bool(hits)))
        dbg["used"] = state.used_user_longterm_memory
        dbg["hits"] = len(hits)

        if not state.used_user_longterm_memory:
            dbg["reason"] = ltm_info.get("reason") or "no_hits_or_not_used"
            state.set_debug_value("user_longterm_memory_hits", 0)
            return

        bundle = state.context.user_longterm_memory_prompt_builder.build_user_longterm_memory_prompt(hits)
        context_messages = bundle.context_messages or []
        insert_context_before_last_user(state, context_messages)

        # Debug trace (no tools coupling)
        ltm_context_texts: List[str] = []
        for msg in context_messages:
            if msg.content:
                ltm_context_texts.append(msg.content)

        dbg["context_blocks_count"] = len(ltm_context_texts)

        preview = "\n\n".join(ltm_context_texts[:2])  # first 1-2 blocks is enough
        dbg["context_preview"] = preview
        dbg["context_preview_chars"] = len(preview)

        dbg["hits_preview"] = [
            {
                "entry_id": h.entry_id,
                "title": getattr(h, "title", None),
                "kind": getattr(getattr(h, "kind", None), "value", getattr(h, "kind", None)),
                "deleted": bool(getattr(h, "deleted", False)),
            }
            for h in hits[:5]
        ]

        state.set_debug_value("user_longterm_memory_hits", len(hits))

        state.trace_event(
            component="engine",
            step="user_longterm_memory",
            message="User long-term memory step executed.",
            data={
                "ltm_enabled": state.context.config.enable_user_longterm_memory,
                "used_user_longterm_memory": state.used_user_longterm_memory,
                "hits": len(hits),
            },
        )