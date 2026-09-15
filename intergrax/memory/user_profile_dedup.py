# © Artur Czarnecki. All rights reserved.

"""LTM dedup and merge policy on consolidate write (Phase MEM-DEPTH-3.2)."""

from __future__ import annotations

from typing import List, Sequence

from intergrax.memory.strategies.defaults.sequence_deduplication import (
    SequenceMatcherDeduplicationConfig,
    SequenceMatcherMemoryDeduplicationStrategy,
)
from intergrax.memory.strategies.models import MemoryDeduplicationRequest
from intergrax.memory.user_profile_memory import UserProfileMemoryEntry


def deduplicate_memory_entries(
    existing: Sequence[UserProfileMemoryEntry],
    incoming: Sequence[UserProfileMemoryEntry],
    *,
    similarity_threshold: float = 0.88,
) -> List[UserProfileMemoryEntry]:
    """Drop near-duplicate incoming entries (classification only; supersession deferred MEM-ENT-6)."""
    strategy = SequenceMatcherMemoryDeduplicationStrategy(
        SequenceMatcherDeduplicationConfig(similarity_threshold=similarity_threshold)
    )
    result = strategy.deduplicate(
        MemoryDeduplicationRequest(
            existing=tuple(existing),
            incoming=tuple(incoming),
        )
    )
    return list(result.accepted)
