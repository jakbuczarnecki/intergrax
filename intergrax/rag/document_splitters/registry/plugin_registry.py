# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Chunking strategy plugin registry (M-RAG.66)."""

from __future__ import annotations

from typing import Callable, Dict

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy
from intergrax.rag.document_splitters.registry.strategy_registry import ChunkingStrategyRegistry

ChunkingStrategyFactory = Callable[[], BaseChunkingStrategy]

_PLUGIN_FACTORIES: Dict[str, ChunkingStrategyFactory] = {}


def register_chunking_strategy_plugin(strategy_id: str, factory: ChunkingStrategyFactory) -> None:
    """Register a third-party chunking strategy factory by stable ``strategy_id``."""
    key = strategy_id.strip().lower()
    if not key:
        raise ValueError("strategy_id must be non-empty")
    _PLUGIN_FACTORIES[key] = factory


def list_chunking_strategy_plugins() -> tuple[str, ...]:
    return tuple(sorted(_PLUGIN_FACTORIES.keys()))


def apply_chunking_strategy_plugins(registry: ChunkingStrategyRegistry) -> int:
    """
    Register all plugin strategies that are not already present.

    Returns the number of strategies registered.
    """
    registered = 0
    for strategy_id, factory in _PLUGIN_FACTORIES.items():
        try:
            registry.resolve(strategy_id)
        except RuntimeError:
            registry.register(factory())
            registered += 1
    return registered
