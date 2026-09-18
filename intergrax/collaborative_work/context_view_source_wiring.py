# © Artur Czarnecki. All rights reserved.

"""Composition root wiring: source readers → MP-5F-B5 adapters → ``DefaultContextViewComposer``."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TypeVar

from intergrax.collaborative_work.context_view_async_reference_read import (
    ContextViewAsyncReferenceReadRunner,
)
from intergrax.collaborative_work.context_view_composition import DefaultContextViewComposer
from intergrax.collaborative_work.context_view_source_adapters import (
    DefaultCollaborativeWorkContextSource,
    DefaultKnowledgeContextSource,
    DefaultMemoryContextSource,
    DefaultUclContextSource,
)
from intergrax.collaborative_work.contracts.collaborative_work_reference_read import (
    CollaborativeWorkReferenceReadPort,
)
from intergrax.contracts.context_view_composition import DefaultContextViewComposerConfig
from intergrax.knowledge.contracts.knowledge_reference_read import KnowledgeReferenceReadPort
from intergrax.memory.contracts.memory_reference_read import MemoryReferenceReadPort
from intergrax.ucl.contracts.ucl_reference_read import UclReferenceReadPort

T = TypeVar("T")

__all__ = [
    "ContextViewSourceIntegration",
    "DefaultContextViewAsyncReferenceReadRunner",
    "wire_default_context_view_composer",
]


@dataclass(frozen=True, slots=True)
class ContextViewSourceIntegration:
    """Explicit DI bundle for ContextView source ports and optional default readers."""

    memory_source: DefaultMemoryContextSource | None = None
    knowledge_source: DefaultKnowledgeContextSource | None = None
    ucl_source: DefaultUclContextSource | None = None
    collaborative_work_source: DefaultCollaborativeWorkContextSource | None = None


class DefaultContextViewAsyncReferenceReadRunner:
    """Platform default async runner for composition roots (not used inside adapters)."""

    def run(self, coro: Coroutine[object, object, T]) -> T:
        if not asyncio.iscoroutine(coro):
            raise TypeError("expected coroutine")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError(
            "DefaultContextViewAsyncReferenceReadRunner cannot run inside an active event loop; "
            "inject a host-specific ContextViewAsyncReferenceReadRunner at composition time",
        )


def wire_default_context_view_composer(
    *,
    memory_reader: MemoryReferenceReadPort | None = None,
    knowledge_reader: KnowledgeReferenceReadPort | None = None,
    ucl_reader: UclReferenceReadPort | None = None,
    collaborative_work_reader: CollaborativeWorkReferenceReadPort | None = None,
    knowledge_reference_read_query_text: str,
    async_runner: ContextViewAsyncReferenceReadRunner | None = None,
    config: DefaultContextViewComposerConfig | None = None,
) -> DefaultContextViewComposer:
    """Instantiate default adapters and inject them as MP-5D ports into the composer."""
    runner = async_runner or DefaultContextViewAsyncReferenceReadRunner()
    memory_source = (
        DefaultMemoryContextSource(reader=memory_reader, async_runner=runner)
        if memory_reader is not None
        else None
    )
    knowledge_source = (
        DefaultKnowledgeContextSource(reader=knowledge_reader)
        if knowledge_reader is not None
        else None
    )
    ucl_source = (
        DefaultUclContextSource(reader=ucl_reader, async_runner=runner)
        if ucl_reader is not None
        else None
    )
    collaborative_work_source = (
        DefaultCollaborativeWorkContextSource(reader=collaborative_work_reader)
        if collaborative_work_reader is not None
        else None
    )
    composer_config = config
    if composer_config is None:
        composer_config = DefaultContextViewComposerConfig(
            knowledge_reference_read_query_text=knowledge_reference_read_query_text,
        )
    return DefaultContextViewComposer(
        memory_source=memory_source,
        knowledge_source=knowledge_source,
        ucl_source=ucl_source,
        collaborative_work_source=collaborative_work_source,
        config=composer_config,
    )
