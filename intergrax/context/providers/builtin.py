# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Shipped builtin context providers — stubs delegating to Nexus as-built (CE-2.3)."""

from __future__ import annotations

from typing import Callable

from intergrax.context.contracts import (
    ContextAssemblyRequest,
    ContextFragment,
    ContextFragmentSource,
    ContextProviderContext,
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
    """Registers all architecture §8.4 builtin providers (stub collect until CE-3)."""

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

        for provider_id, source in _BUILTIN_SPECS:
            if provider_id == "builtin.workspace":
                registry.add_provider(WorkspaceContextProvider())
                continue
            registry.add_provider(_make_stub_provider(provider_id, source))  # type: ignore[arg-type]
        registry.add_provider(SessionSemanticRecallProvider())

    @classmethod
    def builtin_provider_ids(cls) -> tuple[str, ...]:
        return tuple(spec[0] for spec in _BUILTIN_SPECS) + (
            "builtin.session_history_semantic",
        )
