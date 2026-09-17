# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations

from intergrax.contracts.agent_run import RequestIdentity
from intergrax.memory.contracts.memory_control import (
    MemoryControlAccessDenied,
    MemoryControlPlane,
    MemoryControlRecallRequest,
    MemoryControlRememberRequest,
    user_memory_scope,
)
from intergrax.memory.user_profile_memory import MemoryKind
from intergrax.tools._shared.async_dispatch import run_async
from intergrax.tools.providers.ltm.contracts import (
    LtmMemoryHit,
    LtmSearchInput,
    LtmSearchOutput,
    LtmWriteFactInput,
    LtmWriteFactOutput,
)
from intergrax.tools.registry.wiring import ToolWiringContext

LTM_SEARCH_TOOL_ID = "ltm.search"
LTM_WRITE_FACT_TOOL_ID = "ltm.write_fact"


def _require_memory_control_context(
    ctx: ToolWiringContext,
) -> tuple[MemoryControlPlane, RequestIdentity]:
    identity = ctx.extras.get("request_identity")
    if not isinstance(identity, RequestIdentity):
        raise MemoryControlAccessDenied("trusted request identity required")
    if not (identity.user_id or "").strip():
        raise MemoryControlAccessDenied("trusted request identity required")
    plane = ctx.extras.get("memory_control_plane")
    if not isinstance(plane, MemoryControlPlane):
        raise MemoryControlAccessDenied("memory_control_plane_not_configured")
    return plane, identity


def _assert_tool_user_matches_identity(params_user_id: str, identity: RequestIdentity) -> None:
    requested = params_user_id.strip()
    canonical = (identity.user_id or "").strip()
    if requested and canonical and requested != canonical:
        raise MemoryControlAccessDenied("ltm user_id conflicts with canonical RequestIdentity")


def ltm_search(ctx: ToolWiringContext, params: LtmSearchInput) -> LtmSearchOutput:
    plane, identity = _require_memory_control_context(ctx)
    query = params.query.strip()
    _assert_tool_user_matches_identity(params.user_id, identity)
    scope = user_memory_scope(identity)
    recall = run_async(
        plane.recall(
            identity,
            scope,
            MemoryControlRecallRequest(query=query, top_k=params.top_k),
        )
    )
    hits = [
        LtmMemoryHit(
            entry_id=item.entry_id,
            content=item.content,
            kind=item.kind.value,
            score=float(item.score or 0.0),
        )
        for item in recall.items
    ]
    return LtmSearchOutput(
        used=bool(hits),
        hits=hits,
        reason=recall.reason,
    )


def ltm_write_fact(ctx: ToolWiringContext, params: LtmWriteFactInput) -> LtmWriteFactOutput:
    plane, identity = _require_memory_control_context(ctx)
    _assert_tool_user_matches_identity(params.user_id, identity)
    scope = user_memory_scope(identity)
    kind_name = params.kind.strip().lower() or "user_fact"
    try:
        kind = MemoryKind(kind_name)
    except ValueError:
        kind = MemoryKind.OTHER
    remembered = run_async(
        plane.remember(
            identity,
            scope,
            MemoryControlRememberRequest(
                content=params.content.strip(),
                kind=kind,
                title=params.title.strip() or None,
            ),
        )
    )
    return LtmWriteFactOutput(
        written=True,
        entry_id=remembered.entry_id or "",
    )
