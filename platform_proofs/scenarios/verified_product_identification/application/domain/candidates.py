"""Derived retrieval candidate models — recall authority, not source truth."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductIdentifier,
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


class RetrievalChannel(StrEnum):
    """Logical retrieval channel — explicit, not stringly typed."""

    EXACT = "exact"
    LEXICAL = "lexical"
    STRUCTURED = "structured"
    VECTOR = "vector"


@dataclass(frozen=True, slots=True)
class ExactChannelScore:
    """Channel-local exact-match evidence — not probabilistic confidence."""

    matched_identifier: ProductIdentifier


@dataclass(frozen=True, slots=True)
class LexicalChannelScore:
    """Channel-local lexical relevance (for example BM25) — unnormalized semantics."""

    bm25_score: float


@dataclass(frozen=True, slots=True)
class StructuredChannelScore:
    """Channel-local structured constraint satisfaction."""

    matched_constraint_count: int
    total_constraint_count: int

    def __post_init__(self) -> None:
        matched = self.matched_constraint_count
        total = self.total_constraint_count
        if type(matched) is not int or matched < 0:
            raise ValueError("matched_constraint_count must be a non-negative int")
        if type(total) is not int or total < 1:
            raise ValueError("total_constraint_count must be a positive int")
        if matched > total:
            raise ValueError("matched_constraint_count cannot exceed total_constraint_count")


@dataclass(frozen=True, slots=True)
class VectorChannelScore:
    """Channel-local vector similarity — never identity evidence."""

    cosine_similarity: float


ChannelScore = (
    ExactChannelScore | LexicalChannelScore | StructuredChannelScore | VectorChannelScore
)

_CHANNEL_SCORE_TYPES: dict[RetrievalChannel, type[ChannelScore]] = {
    RetrievalChannel.EXACT: ExactChannelScore,
    RetrievalChannel.LEXICAL: LexicalChannelScore,
    RetrievalChannel.STRUCTURED: StructuredChannelScore,
    RetrievalChannel.VECTOR: VectorChannelScore,
}


def _validate_rank(rank: int) -> int:
    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative int")
    return rank


def _validate_channel_score(channel: RetrievalChannel, score: ChannelScore | None) -> None:
    if score is None:
        return
    expected_type = _CHANNEL_SCORE_TYPES[channel]
    if not isinstance(score, expected_type):
        raise TypeError(
            f"{channel.value} candidates require {expected_type.__name__}, "
            f"got {type(score).__name__}"
        )


@dataclass(frozen=True, slots=True)
class ProductCandidate:
    """Lightweight derived candidate — sufficient to reach immutable source truth later."""

    offer_id: ProductOfferId
    channel: RetrievalChannel
    rank: int
    source_ref: SourceRecordRef
    channel_score: ChannelScore | None = None

    def __post_init__(self) -> None:
        rank = _validate_rank(self.rank)
        if rank != self.rank:
            object.__setattr__(self, "rank", rank)
        if self.source_ref.offer_id != self.offer_id:
            raise ValueError("source_ref.offer_id must match candidate offer_id")
        _validate_channel_score(self.channel, self.channel_score)


@dataclass(frozen=True, slots=True)
class ChannelCandidateBatch:
    """Single-channel candidate batch before multi-channel handoff."""

    channel: RetrievalChannel
    candidates: tuple[ProductCandidate, ...]

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        for candidate in candidates:
            if candidate.channel != self.channel:
                raise ValueError("all candidates in a batch must share the batch channel")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class MultiChannelCandidateCollection:
    """Common candidate handoff surface for fusion — preserves per-channel semantics."""

    candidates: tuple[ProductCandidate, ...]

    @classmethod
    def from_channel_batches(
        cls,
        *batches: ChannelCandidateBatch,
    ) -> MultiChannelCandidateCollection:
        merged: list[ProductCandidate] = []
        for batch in batches:
            merged.extend(batch.candidates)
        return cls(candidates=tuple(merged))

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
