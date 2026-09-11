"""Immutable offer-level fusion contracts — ranking utility, not verification."""

from __future__ import annotations

import math
from dataclasses import dataclass

from platform_proofs.scenarios.verified_product_identification.application.domain.candidates import (
    ChannelScore,
    MultiChannelCandidateCollection,
    RetrievalChannel,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)


@dataclass(frozen=True, slots=True)
class OfferFusionConfiguration:
    """RRF configuration — ``rrf_k=60`` is the standard conservative baseline, not VPI-tuned."""

    rrf_k: int = 60

    def __post_init__(self) -> None:
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise ValueError("rrf_k must be a positive int")


@dataclass(frozen=True, slots=True)
class OfferChannelEvidence:
    """Per-channel retrieval evidence preserved through fusion."""

    channel: RetrievalChannel
    channel_rank: int
    channel_score: ChannelScore | None
    reciprocal_rank_contribution: float

    def __post_init__(self) -> None:
        _validate_positive_finite(
            self.reciprocal_rank_contribution,
            field_name="OfferChannelEvidence.reciprocal_rank_contribution",
        )


@dataclass(frozen=True, slots=True)
class FusedOfferCandidate:
    """Fused offer-level candidate — fusion_score ranks offers, it is not identity confidence."""

    source_ref: SourceRecordRef
    offer_id: ProductOfferId
    fused_rank: int
    fusion_score: float
    supporting_channel_count: int
    evidence: tuple[OfferChannelEvidence, ...]

    def __post_init__(self) -> None:
        if type(self.fused_rank) is not int or self.fused_rank < 0:
            raise ValueError("fused_rank must be a non-negative int")
        _validate_positive_finite(self.fusion_score, field_name="FusedOfferCandidate.fusion_score")
        if type(self.supporting_channel_count) is not int or self.supporting_channel_count < 1:
            raise ValueError("supporting_channel_count must be a positive int")
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if len(self.evidence) != self.supporting_channel_count:
            raise ValueError("supporting_channel_count must match evidence length")
        if self.source_ref.offer_id != self.offer_id:
            raise ValueError("source_ref.offer_id must match offer_id")


@dataclass(frozen=True, slots=True)
class FusedOfferCandidateCollection:
    """Deterministic ranked offer list after multi-channel fusion."""

    candidates: tuple[FusedOfferCandidate, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        for rank, candidate in enumerate(self.candidates):
            if candidate.fused_rank != rank:
                raise ValueError("fused ranks must be contiguous from zero")


@dataclass(frozen=True, slots=True)
class OfferCandidateFusionRequest:
    """Validated fusion request — query-independent after retrieval."""

    candidates: MultiChannelCandidateCollection
    limit: int

    def __post_init__(self) -> None:
        if type(self.limit) is not int or self.limit <= 0:
            raise ValueError("limit must be a positive int")


def reciprocal_rank_contribution(*, rank: int, rrf_k: int) -> float:
    """Compute one channel contribution using zero-based rank: ``1 / (rrf_k + rank + 1)``."""

    if type(rank) is not int or rank < 0:
        raise ValueError("rank must be a non-negative int")
    if type(rrf_k) is not int or rrf_k <= 0:
        raise ValueError("rrf_k must be a positive int")
    return 1.0 / (rrf_k + rank + 1)


def _validate_positive_finite(value: float, *, field_name: str) -> None:
    if type(value) not in (int, float):
        raise ValueError(f"{field_name} must be a float or int")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{field_name} must be a finite value greater than zero")
