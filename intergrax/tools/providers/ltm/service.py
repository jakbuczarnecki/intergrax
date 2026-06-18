# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

from __future__ import annotations
from intergrax.utils import attribute_access

from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry
from intergrax.tools._shared.async_dispatch import run_async
from intergrax.tools.providers.ltm.contracts import (
    LtmMemoryHit,
    LtmSearchInput,
    LtmSearchOutput,
    LtmWriteFactInput,
    LtmWriteFactOutput,
)
from intergrax.tools.registry.runtime_bindings import UserProfileManagerBinding
from intergrax.tools.registry.wiring import ToolWiringContext

LTM_SEARCH_TOOL_ID = "ltm.search"
LTM_WRITE_FACT_TOOL_ID = "ltm.write_fact"


def _require_user_profile_manager(ctx: ToolWiringContext) -> UserProfileManagerBinding:
    manager = ctx.user_profile_manager or ctx.extras.get("user_profile_manager")
    if manager is None:
        raise RuntimeError("user_profile_manager_not_configured")
    if not isinstance(manager, UserProfileManagerBinding):
        raise RuntimeError("user_profile_manager_invalid_type")
    return manager


def _keyword_hits(profile: object, query: str) -> list[LtmMemoryHit]:
    needle = query.lower()
    hits: list[LtmMemoryHit] = []
    entries = attribute_access.optional(profile, "memory_entries", []) or []
    for entry in entries:
        if attribute_access.optional(entry, "deleted", False):
            continue
        content = str(attribute_access.optional(entry, "content", "") or "")
        if needle in content.lower():
            kind = attribute_access.optional(entry, "kind", MemoryKind.OTHER)
            kind_value = kind.value if hasattr(kind, "value") else str(kind)
            hits.append(
                LtmMemoryHit(
                    entry_id=str(attribute_access.optional(entry, "entry_id", "")),
                    content=content,
                    kind=kind_value,
                    score=1.0,
                )
            )
    return hits


def ltm_search(ctx: ToolWiringContext, params: LtmSearchInput) -> LtmSearchOutput:
    manager = _require_user_profile_manager(ctx)
    user_id = params.user_id.strip()
    query = params.query.strip()

    if manager.is_longterm_rag_enabled():
        result = run_async(
            manager.search_longterm_memory(
                user_id,
                query,
                top_k=params.top_k,
            )
        )
        hits_raw = result.get("hits") or []
        scores = result.get("scores") or []
        hits: list[LtmMemoryHit] = []
        for index, entry in enumerate(hits_raw):
            kind = attribute_access.optional(entry, "kind", MemoryKind.OTHER)
            kind_value = kind.value if hasattr(kind, "value") else str(kind)
            score = float(scores[index]) if index < len(scores) else 0.0
            hits.append(
                LtmMemoryHit(
                    entry_id=str(attribute_access.optional(entry, "entry_id", "")),
                    content=str(attribute_access.optional(entry, "content", "") or ""),
                    kind=kind_value,
                    score=score,
                )
            )
        debug = result.get("debug") or {}
        used = bool(debug.get("used") or result.get("used_longterm"))
        return LtmSearchOutput(used=used, hits=hits, reason=str(debug.get("reason") or "ok"))

    profile = run_async(manager.get_profile(user_id))
    hits = _keyword_hits(profile, query)[: params.top_k]
    return LtmSearchOutput(used=bool(hits), hits=hits, reason="keyword_fallback")


def ltm_write_fact(ctx: ToolWiringContext, params: LtmWriteFactInput) -> LtmWriteFactOutput:
    manager = _require_user_profile_manager(ctx)
    kind_name = params.kind.strip().lower() or "user_fact"
    try:
        kind = MemoryKind(kind_name)
    except ValueError:
        kind = MemoryKind.OTHER
    entry = UserProfileMemoryEntry(
        content=params.content.strip(),
        kind=kind,
        title=params.title.strip() or None,
    )
    saved = run_async(manager.add_memory_entry(params.user_id.strip(), entry))
    return LtmWriteFactOutput(written=True, entry_id=str(attribute_access.optional(saved, "entry_id", "")))
