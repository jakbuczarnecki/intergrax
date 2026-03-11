# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from intergrax.rag.indexing.contracts.index_strategy import IndexStrategy
from intergrax.rag.indexing.pipeline.indexing_pipeline import IndexingPipeline
from intergrax.rag.indexing.strategies.single_index_strategy import SingleIndexStrategy


def create_default_index_strategy(
    *,
    strategy: IndexStrategy | None = None,
) -> IndexStrategy:
    """
    Create default indexing strategy.

    Allows dependency override by providing a custom strategy.
    """

    if strategy is None:
        strategy = SingleIndexStrategy()

    return strategy


def create_default_indexing_pipeline(
    *,
    strategy: IndexStrategy | None = None,
) -> IndexingPipeline:
    """
    Create IndexingPipeline using the default indexing strategy.
    """

    if strategy is None:
        strategy = create_default_index_strategy()

    return IndexingPipeline(
        strategy=strategy,
    )