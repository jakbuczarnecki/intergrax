# © Artur Czarnecki. All rights reserved.

"""LTM dedup and merge policy on consolidate write (Phase MEM-DEPTH-3.2)."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Sequence

from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


def deduplicate_memory_entries(
    existing: Sequence[UserProfileMemoryEntry],
    incoming: Sequence[UserProfileMemoryEntry],
    *,
    similarity_threshold: float = 0.88,
) -> List[UserProfileMemoryEntry]:
    """
    Drop near-duplicate incoming entries; supersede stale facts when newer arrives.
    """
    accepted: List[UserProfileMemoryEntry] = []
    active_existing = [entry for entry in existing if not entry.deleted]

    for candidate in incoming:
        if candidate.deleted:
            continue
        duplicate = False
        for prior in active_existing + accepted:
            if prior.kind != candidate.kind:
                continue
            if _similarity(prior.content, candidate.content) >= similarity_threshold:
                duplicate = True
                if candidate.kind in {MemoryKind.USER_FACT, MemoryKind.PREFERENCE}:
                    prior.valid_until = candidate.created_at or prior.valid_until
                break
        if not duplicate:
            accepted.append(candidate)
    return accepted
