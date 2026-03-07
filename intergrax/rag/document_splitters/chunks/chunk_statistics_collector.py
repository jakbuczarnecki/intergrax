# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from typing import List

from intergrax.rag.document_splitters.contracts.chunk_statistics import (
    ChunkStatistics,
)


class ChunkStatisticsCollector:
    """
    Collects statistics about the chunking process.
    """

    def __init__(self) -> None:
        self._sizes: List[int] = []

    def observe(self, text: str) -> None:
        """
        Register a chunk for statistics collection.
        """
        self._sizes.append(len(text))

    def build(self) -> ChunkStatistics:
        """
        Build statistics describing the chunking process.
        """

        if not self._sizes:
            return ChunkStatistics(
                chunk_count=0,
                avg_chunk_size=0.0,
                min_chunk_size=0,
                max_chunk_size=0,
            )

        chunk_count = len(self._sizes)

        avg_size = sum(self._sizes) / chunk_count
        min_size = min(self._sizes)
        max_size = max(self._sizes)

        return ChunkStatistics(
            chunk_count=chunk_count,
            avg_chunk_size=avg_size,
            min_chunk_size=min_size,
            max_chunk_size=max_size,
        )