# © Artur Czarnecki. All rights reserved.

"""Default sequence-matcher deduplication strategy (MEM-ENT-4)."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from intergrax.memory.strategies.models import (
    MemoryDeduplicationRequest,
    MemoryDeduplicationResult,
)
from intergrax.memory.user_profile_memory import MemoryKind, UserProfileMemoryEntry


@dataclass(frozen=True)
class SequenceMatcherDeduplicationConfig:
    similarity_threshold: float = 0.88


def _similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left.lower().strip(), right.lower().strip()).ratio()


class SequenceMatcherMemoryDeduplicationStrategy:
    strategy_id = "builtin.memory.deduplication.sequence_matcher"

    def __init__(self, config: SequenceMatcherDeduplicationConfig | None = None) -> None:
        self._config = config or SequenceMatcherDeduplicationConfig()

    @property
    def similarity_threshold(self) -> float:
        return self._config.similarity_threshold

    def deduplicate(self, request: MemoryDeduplicationRequest) -> MemoryDeduplicationResult:
        accepted: list[UserProfileMemoryEntry] = []
        rejected: list[UserProfileMemoryEntry] = []
        threshold = self._config.similarity_threshold
        active_existing = [entry for entry in request.existing if not entry.deleted]

        for candidate in request.incoming:
            if candidate.deleted:
                continue
            duplicate = False
            for prior in active_existing + accepted:
                if prior.kind != candidate.kind:
                    continue
                if _similarity(prior.content, candidate.content) >= threshold:
                    duplicate = True
                    if candidate.kind in {MemoryKind.USER_FACT, MemoryKind.PREFERENCE}:
                        prior.valid_until = candidate.created_at or prior.valid_until
                    break
            if duplicate:
                rejected.append(candidate)
            else:
                accepted.append(candidate)

        return MemoryDeduplicationResult(
            accepted=tuple(accepted),
            rejected_as_duplicate=tuple(rejected),
        )
