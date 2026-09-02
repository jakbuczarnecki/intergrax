# © Artur Czarnecki. All rights reserved.

"""Typed web-search candidate models for Q3 qualification."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.websearch.schemas.search_hit import SearchHit


@dataclass(frozen=True, slots=True)
class WebSearchCandidate:
    rank: int
    url: str
    title: str
    snippet: str
    provider: str


def candidates_from_hits(hits: tuple[SearchHit, ...]) -> tuple[WebSearchCandidate, ...]:
    candidates: list[WebSearchCandidate] = []
    for hit in hits:
        candidates.append(
            WebSearchCandidate(
                rank=hit.rank,
                url=hit.url,
                title=hit.title,
                snippet=(hit.snippet or "")[:500],
                provider=hit.provider,
            ),
        )
    return tuple(candidates)


__all__ = ["WebSearchCandidate", "candidates_from_hits"]
