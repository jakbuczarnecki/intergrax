# © Artur Czarnecki. All rights reserved.

"""LTM and episodic recall population for CE provider handles (MEM-VEC-2.3)."""

from __future__ import annotations

from typing import Any

from intergrax.runtime.nexus.config import RuntimeConfig
from intergrax.runtime.nexus.context.context_handle_rows import ltm_entry_row
from intergrax.runtime.nexus.context.provider_handles import (
    LTM_ENTRIES_METADATA_KEY,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState
from intergrax.runtime.nexus.responses.response_schema import RuntimeRequest
from intergrax.runtime.nexus.session.session_manager import SessionManager
from intergrax.runtime.nexus.tracing.memory.user_longterm_memory_summary import (
    UserLongtermMemorySummaryDiagV1,
)
from intergrax.runtime.nexus.tracing.trace_models import TraceComponent, TraceLevel


def memory_profile_handle_snapshot(config: RuntimeConfig) -> dict[str, Any]:
    return {
        "enable_session_vector_index": config.enable_session_vector_index,
        "include_cross_session_episodic": config.include_cross_session_episodic,
        "session_index_top_k": config.session_index_top_k,
    }


async def populate_request_memory_recall_metadata(
    request: RuntimeRequest,
    *,
    config: RuntimeConfig,
    session_manager: SessionManager,
) -> None:
    """Fill ``request.metadata`` with LTM + episodic hits for CE providers."""
    request.metadata["memory_profile"] = memory_profile_handle_snapshot(config)
    query = (request.message or "").strip()
    if not query:
        return

    user_id = str(request.user_id or request.metadata.get("user_id") or "")
    tenant_id = str(request.tenant_id or request.metadata.get("tenant_id") or "default")
    session_id = str(request.session_id or request.metadata.get("session_id") or "")

    if config.enable_user_longterm_memory and user_id:
        result = await session_manager.search_user_longterm_memory(
            user_id,
            query,
            top_k=config.max_longterm_entries_per_query,
            score_threshold=config.longterm_score_threshold,
        )
        if result and result.get("hits"):
            request.metadata[LTM_ENTRIES_METADATA_KEY] = [
                ltm_entry_row(entry) for entry in result["hits"]
            ]

    if config.enable_session_vector_index and session_id:
        hits = await session_manager.search_session_semantic_recall(
            tenant_id=tenant_id,
            session_id=session_id,
            user_id=user_id or None,
            query=query,
            top_k=config.session_index_top_k,
            score_threshold=config.session_index_score_threshold,
        )
        if hits:
            request.metadata["session_vector_hits"] = hits


async def run_longterm_memory_context(state: RuntimeState) -> None:
    """Retrieve LTM hits into runtime state for CE bridge + legacy injection."""
    cfg = state.context.config
    state.used_user_longterm_memory = False
    if not cfg.enable_user_longterm_memory:
        return

    session_manager = state.context.session_manager
    user_id = str(state.request.user_id or state.request.metadata.get("user_id") or "")
    query = (state.request.message or "").strip()
    if not user_id or not query:
        return

    result = await session_manager.search_user_longterm_memory(
        user_id,
        query,
        top_k=cfg.max_longterm_entries_per_query,
        score_threshold=cfg.longterm_score_threshold,
    )
    if not result:
        return

    state.user_longterm_memory_result = result
    used = bool(result.get("used_longterm") or result.get("debug", {}).get("used"))
    state.used_user_longterm_memory = used
    context_blocks_count = 0
    if used and state.context.user_longterm_memory_prompt_builder is not None:
        hits = result.get("hits") or []
        bundle = state.context.user_longterm_memory_prompt_builder.build_user_longterm_memory_prompt(
            hits,
        )
        context_blocks_count = len(bundle.context_messages)
        if bundle.context_messages:
            from intergrax.runtime.nexus.context.tool_context_helpers import (
                insert_context_before_last_user,
            )

            insert_context_before_last_user(state, bundle.context_messages)

    state.trace_event(
        component=TraceComponent.ENGINE,
        step="longterm_memory",
        message="Long-term memory retrieval completed.",
        level=TraceLevel.INFO,
        payload=UserLongtermMemorySummaryDiagV1(
            enabled=True,
            used_user_longterm_memory=used,
            reason=str(result.get("debug", {}).get("reason") or ""),
            hits_count=len(result.get("hits") or []),
            top_k=cfg.max_longterm_entries_per_query,
            context_blocks_count=context_blocks_count,
            context_preview_chars=0,
            context_preview="",
        ),
    )


async def run_session_semantic_recall_context(state: RuntimeState) -> None:
    """Populate episodic vector hits on request metadata for CE providers."""
    cfg = state.context.config
    if not cfg.enable_session_vector_index:
        return

    session_manager = state.context.session_manager
    query = (state.request.message or "").strip()
    tenant_id = str(state.request.tenant_id or "default")
    session_id = str(state.request.session_id or "")
    user_id = str(state.request.user_id or state.request.metadata.get("user_id") or "") or None
    if not query or not session_id:
        return

    hits = await session_manager.search_session_semantic_recall(
        tenant_id=tenant_id,
        session_id=session_id,
        user_id=user_id,
        query=query,
        top_k=cfg.session_index_top_k,
        score_threshold=cfg.session_index_score_threshold,
    )
    state.request.metadata["memory_profile"] = memory_profile_handle_snapshot(cfg)
    if hits:
        state.request.metadata["session_vector_hits"] = hits
