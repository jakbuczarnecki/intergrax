# © Artur Czarnecki. All rights reserved.

"""LTM and episodic recall population for CE provider handles (MEM-VEC-2.3)."""

from __future__ import annotations

from typing import Any

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.contracts.request_identity_spine import (
    verified_request_identity_for_memory_recall,
)
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlPlane,
    MemoryControlRecallRequest,
    MemoryControlRecallResult,
    user_memory_scope,
)
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry
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
from intergrax.runtime.task_memory.metrics import memory_platform_metrics


def _trusted_recall_identity(request: RuntimeRequest) -> RequestIdentity | None:
    return verified_request_identity_for_memory_recall(
        request.canonical_identity,
        metadata=request.metadata,
        legacy_tenant_id=request.tenant_id,
    )


def memory_control_plane_from_config(config: RuntimeConfig) -> MemoryControlPlane | None:
    wiring = config.tool_wiring_context
    if wiring is None:
        return None
    plane = wiring.extras.get("memory_control_plane")
    if isinstance(plane, MemoryControlPlane):
        return plane
    return None


def _require_memory_control_plane(config: RuntimeConfig) -> MemoryControlPlane:
    plane = memory_control_plane_from_config(config)
    if plane is None:
        raise MemoryControlAccessDenied("memory_control_plane_not_configured")
    return plane


def engine_ltm_recall_from_control_plane(recall: MemoryControlRecallResult) -> dict[str, Any]:
    hits: list[UserProfileMemoryEntry] = []
    scores: list[float] = []
    for item in recall.items:
        hits.append(
            UserProfileMemoryEntry(
                entry_id=item.entry_id,
                content=item.content,
                kind=item.kind,
            )
        )
        scores.append(float(item.score or 0.0))
    used = bool(hits)
    reason = recall.reason or ("hits" if used else "no_hits")
    return {
        "used_longterm": used,
        "hits": hits,
        "scores": scores,
        "debug": {
            "enabled": True,
            "used": used,
            "reason": reason,
            "hits_count": len(hits),
            "used_semantic": recall.used_semantic,
        },
    }


async def recall_durable_user_memory_via_plane(
    *,
    config: RuntimeConfig,
    identity: RequestIdentity,
    query: str,
    top_k: int,
    score_threshold: float | None,
) -> dict[str, Any]:
    plane = _require_memory_control_plane(config)
    scope = user_memory_scope(identity)
    recall = await plane.recall(
        identity,
        scope,
        MemoryControlRecallRequest(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
        ),
    )
    return engine_ltm_recall_from_control_plane(recall)


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

    identity = _trusted_recall_identity(request)
    if identity is None:
        return

    user_id = identity.user_id or ""
    tenant_id = identity.tenant_id
    session_id = str(request.session_id or request.metadata.get("session_id") or "")

    if config.enable_user_longterm_memory and user_id:
        result = await recall_durable_user_memory_via_plane(
            config=config,
            identity=identity,
            query=query,
            top_k=config.max_longterm_entries_per_query,
            score_threshold=config.longterm_score_threshold,
        )
        if result.get("hits"):
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

    identity = _trusted_recall_identity(state.request)
    user_id = (identity.user_id or "") if identity is not None else ""
    query = (state.request.message or "").strip()
    if identity is None or not user_id or not query:
        return

    result = await recall_durable_user_memory_via_plane(
        config=cfg,
        identity=identity,
        query=query,
        top_k=cfg.max_longterm_entries_per_query,
        score_threshold=cfg.longterm_score_threshold,
    )
    if not result:
        return

    state.user_longterm_memory_result = result
    used = bool(result.get("used_longterm") or result.get("debug", {}).get("used"))
    state.used_user_longterm_memory = used
    if used:
        memory_platform_metrics().record_ltm_hit()
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
    identity = _trusted_recall_identity(state.request)
    if identity is None:
        return

    query = (state.request.message or "").strip()
    tenant_id = identity.tenant_id
    session_id = str(state.request.session_id or "")
    user_id = identity.user_id or None
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
        memory_platform_metrics().record_episodic_hit()
        state.request.metadata["session_vector_hits"] = hits
    elif cfg.enable_session_vector_index:
        state.request.metadata["session_vector_recall_reason"] = "no_hits"
