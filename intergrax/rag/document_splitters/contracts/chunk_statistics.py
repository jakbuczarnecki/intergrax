# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkStatistics:
    """
    Statistics describing the chunking process.
    """

    chunk_count: int
    avg_chunk_size: float
    min_chunk_size: int
    max_chunk_size: int