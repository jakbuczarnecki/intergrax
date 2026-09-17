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
    ContextProviderDescriptor,
)
from intergrax.context.contracts import BUILTIN_PROVIDER_VERSION
from intergrax.context.provider_descriptor import build_provider_descriptor
from intergrax.context.registry import ContextPluginRegistry
from intergrax.context.session_history import (
    HandleSessionHistoryProvider,
    fragments_from_session_history_snapshot,
)
from intergrax.context.source_fragments import (
    fragments_from_attachment_input,
    fragments_from_graph_prior_input,
    fragments_from_memory_input,
    fragments_from_policy_overlay_input,
    fragments_from_rag_input,
    fragments_from_shared_context_input,
    fragments_from_system_input,
    fragments_from_task_message,
    fragments_from_tool_input,
    fragments_from_web_input,
)
from intergrax.llm.messages import ChatMessage

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


def _auxiliary_messages(ctx: ContextProviderContext) -> tuple[ChatMessage, ...] | None:
    if ctx.runtime is not None and ctx.runtime.base_messages:
        return ctx.runtime.base_messages
    messages = ctx.handles.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    typed: list[ChatMessage] = []
    for message in messages:
        if isinstance(message, ChatMessage):
            typed.append(message)
    return tuple(typed) if typed else None


async def _collect_task_message(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    return fragments_from_task_message(request, messages=_auxiliary_messages(ctx))


async def _collect_graph_prior(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    records = ctx.sources.graph_prior
    if not records:
        return []
    max_entries = request.assembly_options.max_prior_entries
    return fragments_from_graph_prior_input(records, max_entries=max_entries)


async def _collect_session_history(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if not request.decision_profile.include_session_history:
        return []
    provider = HandleSessionHistoryProvider()
    snapshot = await provider.load_snapshot(request, ctx)
    if snapshot is None:
        return []
    return fragments_from_session_history_snapshot(snapshot)


async def _collect_rag(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.RAG in request.excluded_sources:
        return []
    if not request.decision_profile.prefer_rag_when_enabled:
        return []
    chunks = ctx.sources.rag
    if not chunks:
        return []
    return fragments_from_rag_input(chunks)


async def _collect_longterm_memory(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.LONGTERM_MEMORY in request.excluded_sources:
        return []
    if not request.decision_profile.prefer_longterm_memory:
        return []
    entries = ctx.sources.memory
    if not entries:
        return []
    max_entries = request.decision_profile.max_memory_entries_in_context
    return fragments_from_memory_input(entries, max_entries=max_entries)


async def _collect_websearch(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.WEBSEARCH in request.excluded_sources:
        return []
    blocks = ctx.sources.web
    if not blocks:
        return []
    return fragments_from_web_input(blocks)


async def _collect_tool_output(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.TOOL_OUTPUT in request.excluded_sources:
        return []
    blocks = ctx.sources.tools
    if not blocks:
        return []
    return fragments_from_tool_input(blocks)


async def _collect_system_instructions(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    system = ctx.sources.system
    if system is None:
        return []
    return fragments_from_system_input(system)


async def _collect_shared_context(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    reads = ctx.sources.shared_context
    if not reads:
        return []
    return fragments_from_shared_context_input(reads)


async def _collect_attachments(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    if ContextFragmentSource.ATTACHMENT in request.excluded_sources:
        return []
    summaries = ctx.sources.attachments
    if not summaries:
        return []
    return fragments_from_attachment_input(summaries)


async def _collect_policy_overlay(
    request: ContextAssemblyRequest,
    ctx: ContextProviderContext,
) -> list[ContextFragment]:
    _ = request
    overlays = ctx.sources.policy_overlay
    if not overlays:
        return []
    return fragments_from_policy_overlay_input(overlays)


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
    provider_version: str = BUILTIN_PROVIDER_VERSION,
) -> object:
    async def _default_collect(
        request: ContextAssemblyRequest,
        ctx: ContextProviderContext,
    ) -> list[ContextFragment]:
        return []

    collect = collect_fn or _default_collect
    normalized_id = provider_id.strip().lower()
    supported_sources = frozenset({source})
    descriptor = build_provider_descriptor(
        normalized_id,
        provider_version=provider_version,
        supported_sources=supported_sources,
        origin="builtin",
    )

    class _StubProvider:
        def __init__(self) -> None:
            self._provider_id = normalized_id
            self._supported_sources = supported_sources
            self._descriptor = descriptor

        @property
        def provider_id(self) -> str:
            return self._provider_id

        @property
        def supported_sources(self) -> frozenset[ContextFragmentSource]:
            return self._supported_sources

        @property
        def descriptor(self) -> ContextProviderDescriptor:
            return self._descriptor

        async def collect(
            self,
            request: ContextAssemblyRequest,
            ctx: ContextProviderContext,
        ) -> list[ContextFragment]:
            return await collect(request, ctx)

    return _StubProvider()


class BuiltinContextPlugin:
    """Registers all architecture §8.4 builtin providers (typed source collectors)."""

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
