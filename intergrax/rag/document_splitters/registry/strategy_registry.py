# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import Dict, Iterable

from intergrax.rag.document_splitters.contracts.base_chunking_strategy import BaseChunkingStrategy


class ChunkingStrategyRegistry:
    """
    Registry responsible for storing and resolving chunking strategies.

    Strategies are registered by their stable strategy_id and can be resolved
    deterministically by name.
    """

    def __init__(
        self,
        strategies: Iterable[BaseChunkingStrategy] | None = None,
    ) -> None:
        self._strategies: Dict[str, BaseChunkingStrategy] = {}

        if strategies is not None:
            for strategy in strategies:
                self.register(strategy)

    def register(
        self,
        strategy: BaseChunkingStrategy,
    ) -> None:
        """
        Register a chunking strategy.

        Raises
        ------
        ValueError
            If strategy with the same strategy_id is already registered.
        """
        strategy_id = strategy.strategy_id()

        if strategy_id in self._strategies:
            raise ValueError(
                f"Chunking strategy already registered: {strategy_id}"
            )

        self._strategies[strategy_id] = strategy

    def resolve(
        self,
        strategy_id: str,
    ) -> BaseChunkingStrategy:
        """
        Resolve a chunking strategy by its stable identifier.

        Raises
        ------
        RuntimeError
            If strategy is not registered.
        """
        strategy = self._strategies.get(strategy_id)

        if strategy is None:
            raise RuntimeError(
                f"Chunking strategy not registered: {strategy_id}"
            )

        return strategy