# © Artur Czarnecki. All rights reserved.

"""MEM-XINT-4: fail-closed guard for canonical Context Engine composition."""

from __future__ import annotations

from intergrax.runtime.nexus.context.provider_handles import (
    LTM_ENTRIES_METADATA_KEY,
    RAG_CHUNKS_METADATA_KEY,
    TOOL_OUTPUT_BLOCKS_METADATA_KEY,
    WEBSEARCH_BLOCKS_METADATA_KEY,
)
from intergrax.runtime.nexus.context.runtime_state_handle_bridge import (
    extract_provider_metadata_from_runtime_state,
)
from intergrax.runtime.nexus.engine.runtime_state import RuntimeState

_ACTIVE_SOURCE_METADATA_KEYS = frozenset(
    {
        LTM_ENTRIES_METADATA_KEY,
        RAG_CHUNKS_METADATA_KEY,
        WEBSEARCH_BLOCKS_METADATA_KEY,
        TOOL_OUTPUT_BLOCKS_METADATA_KEY,
    }
)


def enforce_context_engine_when_provider_sources_active(state: RuntimeState) -> None:
    """Require ``context_engine`` when LTM/RAG/websearch/tool CE handles are populated."""
    if state.context.config.context_engine is not None:
        return
    extracted = extract_provider_metadata_from_runtime_state(state)
    if not any(key in extracted for key in _ACTIVE_SOURCE_METADATA_KEYS):
        return
    raise RuntimeError(
        "context_engine is required for canonical model context composition "
        "when memory, RAG, websearch, or tool output sources are active"
    )
