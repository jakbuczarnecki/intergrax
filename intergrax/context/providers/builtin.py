# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shipped builtin context providers — catalog + live collectors (CE-2.3, CE-PROV-WIRE)."""

from __future__ import annotations

from typing import Callable

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
)
from intergrax.context.providers.legacy_bridge import (
    ATTACHMENT_SUMMARIES_HANDLE,
    LTM_ENTRIES_HANDLE,
    POLICY_OVERLAY_FRAGMENTS_HANDLE,
    PRIOR_OUTPUT_RECORDS_HANDLE,
    RAG_CHUNKS_HANDLE,
    SHARED_CONTEXT_READS_HANDLE,
    SYSTEM_INSTRUCTIONS_HANDLE,
    TOOL_OUTPUT_BLOCKS_HANDLE,
    WEBSEARCH_BLOCKS_HANDLE,
    fragments_from_attachment_summaries,
    fragments_from_ltm_entries,
    fragments_from_policy_overlay_fragments,
    fragments_from_prior_output_records,
    fragments_from_rag_chunks,
    fragments_from_shared_context_reads,
    fragments_from_system_instructions,
    fragments_from_task_message,
    fragments_from_tool_output_blocks,
    fragments_from_websearch_blocks,
)
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    fragments_from_session_history_snapshot,
)
from intergrax.context.registry import ContextPluginRegistry

_BUILTIN_SPECS: tuple[tuple[str, ContextFragmentSource], ...] = (
    ("builtin.task_message", ContextFragmentSource.TASK_MESSAGE),
    ("builtin.system_instructions", ContextFragmentSource.SYSTEM_INSTRUCTIONS),
    ("builtin.session_history", ContextFragmentSource.SESSION_HISTORY),
    ("builtin.longterm_memory", ContextFragmentSource.LONGTERM_MEMORY),
    ("builtin.rag", ContextFragmentSource.RAG),
    ("builtin.websearch", ContextFragmentSource.WEBSEARCH),
    ("builtin.tool_output", ContextFragmentSource.TOOL_OUTPUT),
    ("builtin.graph_prior", ContextFragmentSource.GRAPH_PRIOR),
    ("builtin.shared_context", ContextFragmentSource.SHARED_CONTEXT),
    ("builtin.attachments", ContextFragmentSource.ATTACHMENT),
    ("builtin.policy_overlay", ContextFragmentSource.POLICY_OVERLAY),
    ("builtin.workspace", ContextFragmentSource.WORKSPACE),
)

# CE-PROV-GATE: every catalog builtin except workspace/session_semantic must wire collect.
WIRED_BUILTIN_COLLECTOR_IDS: frozenset[str] = frozenset(
    spec[0] for spec in _BUILTIN_SPECS if spec[0] != "builtin.workspace"
) | frozenset({"builtin.session_history_semantic"})


async def _collect_task_message(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    messages = ctx.handles.get("messages")
    typed_messages = list(messages) if isinstance(messages, list) else None
    return fragments_from_task_message(request, messages=typed_messages)


async def _collect_graph_prior(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    records = ctx.handles.get(PRIOR_OUTPUT_RECORDS_HANDLE)
    if not isinstance(records, list) or not records:
        return []
    max_entries = request.assembly_options.max_prior_entries
    return fragments_from_prior_output_records(records, max_entries=max_entries)


async def _collect_session_history(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    provider = HandleSessionHistoryProvider()
    snapshot = await provider.load_snapshot(request, ctx)
    if snapshot is not None:
        return fragments_from_session_history_snapshot(snapshot)

    from intergrax.context.providers.legacy_bridge import (
        SESSION_HISTORY_MESSAGES_HANDLE,
        fragments_from_session_history,
    )

    raw = ctx.handles.get(SESSION_HISTORY_MESSAGES_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    max_entries = request.decision_profile.max_memory_entries_in_context
    return fragments_from_session_history(raw, max_entries=max_entries)


async def _collect_rag(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.RAG in request.excluded_sources:
        return []
    if not request.decision_profile.prefer_rag_when_enabled:
        return []
    raw = ctx.handles.get(RAG_CHUNKS_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    return fragments_from_rag_chunks(raw)


async def _collect_longterm_memory(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.LONGTERM_MEMORY in request.excluded_sources:
        return []
    if not request.decision_profile.prefer_longterm_memory:
        return []
    raw = ctx.handles.get(LTM_ENTRIES_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    max_entries = request.decision_profile.max_memory_entries_in_context
    return fragments_from_ltm_entries(raw, max_entries=max_entries)


async def _collect_websearch(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.WEBSEARCH in request.excluded_sources:
        return []
    raw = ctx.handles.get(WEBSEARCH_BLOCKS_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    return fragments_from_websearch_blocks(raw)


async def _collect_tool_output(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.TOOL_OUTPUT in request.excluded_sources:
        return []
    raw = ctx.handles.get(TOOL_OUTPUT_BLOCKS_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    return fragments_from_tool_output_blocks(raw)


async def _collect_system_instructions(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    raw = ctx.handles.get(SYSTEM_INSTRUCTIONS_HANDLE)
    if not isinstance(raw, str) or not raw.strip():
        return []
    return fragments_from_system_instructions(raw)


async def _collect_shared_context(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    raw = ctx.handles.get(SHARED_CONTEXT_READS_HANDLE)
    if not isinstance(raw, dict) or not raw:
        return []
    return fragments_from_shared_context_reads(raw)


async def _collect_attachments(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.ATTACHMENT in request.excluded_sources:
        return []
    raw = ctx.handles.get(ATTACHMENT_SUMMARIES_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    return fragments_from_attachment_summaries(raw)


async def _collect_policy_overlay(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    raw = ctx.handles.get(POLICY_OVERLAY_FRAGMENTS_HANDLE)
    if not isinstance(raw, list) or not raw:
        return []
    return fragments_from_policy_overlay_fragments(raw)


_COLLECT_OVERRIDES: dict[str, Callable[..., list[ContextFragment]]] = {
    "builtin.task_message": _collect_task_message,
    "builtin.graph_prior": _collect_graph_prior,
    "builtin.session_history": _collect_session_history,
    "builtin.rag": _collect_rag,
    "builtin.longterm_memory": _collect_longterm_memory,
    "builtin.websearch": _collect_websearch,
    "builtin.tool_output": _collect_tool_output,
    "builtin.system_instructions": _collect_system_instructions,
    "builtin.shared_context": _collect_shared_context,
    "builtin.attachments": _collect_attachments,
    "builtin.policy_overlay": _collect_policy_overlay,
}


def _make_stub_provider(
    provider_id: str,
    source: ContextFragmentSource,
    *,
    collect_fn: Callable[
        [ContextAssemblyRequest, ContextProviderContext],
        list[ContextFragment],
    ]
    | None = None,
) -> object:
    async def _default_collect(
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        return []

    collect = collect_fn or _default_collect

    class _StubProvider:
        def __init__(self) -> None:
            self._provider_id = provider_id
            self._supported_sources = frozenset({source})

        @property
        def provider_id(self) -> str:
            return self._provider_id

        @property
        def supported_sources(self) -> frozenset[ContextFragmentSource]:
            return self._supported_sources

        async def collect(
            self,
            request: ContextAssemblyRequest,
            ctx: ContextProviderContext,
        ) -> list[ContextFragment]:
            return await collect(request, ctx)

    return _StubProvider()


class BuiltinContextPlugin:
    """Registers all architecture §8.4 builtin providers (live collectors via legacy bridge)."""

    @classmethod
    def plugin_id(cls) -> str:
        return "intergrax.builtin"

    @classmethod
    def plugin_version(cls) -> str:
        return "1.0.0"

    @classmethod
    def plugin_description(cls) -> str:
        return "Shipped Intergrax builtin context source providers"

    @classmethod
    def register(cls, registry: ContextPluginRegistry) -> None:
        from intergrax.context.providers.session_semantic_recall import (
            SessionSemanticRecallProvider,
        )
        from intergrax.context.providers.workspace import WorkspaceContextProvider

        from intergrax.context.formatter import DefaultContextFormatter

        registry.set_formatter(DefaultContextFormatter())
        for provider_id, source in _BUILTIN_SPECS:
            if provider_id == "builtin.workspace":
                registry.add_provider(WorkspaceContextProvider())
                continue
            collect_override = _COLLECT_OVERRIDES.get(provider_id)
            registry.add_provider(
                _make_stub_provider(provider_id, source, collect_fn=collect_override)  # type: ignore[arg-type]
            )
        registry.add_provider(SessionSemanticRecallProvider())

    @classmethod
    def builtin_provider_ids(cls) -> tuple[str, ...]:
        return tuple(spec[0] for spec in _BUILTIN_SPECS) + (
            "builtin.session_history_semantic",
        )
