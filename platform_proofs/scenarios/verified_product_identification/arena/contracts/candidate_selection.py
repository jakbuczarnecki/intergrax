"""Typed candidate subset selection for arena execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EmbeddingArenaCandidateSelection:
    """Explicit finalist/challenger subset — composition boundary only."""

    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_ids:
            msg = "candidate_ids must not be empty"
            raise ValueError(msg)
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            msg = "candidate_ids must be unique"
            raise ValueError(msg)
