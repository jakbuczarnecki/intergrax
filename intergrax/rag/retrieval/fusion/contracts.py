# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.

"""Platform-owned rank fusion contracts (vendor- and scenario-neutral)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from intergrax.rag.retrieval.fusion.errors import RankFusionContractError

TPayload = TypeVar("TPayload")

RECIPROCAL_RANK_FUSION_STRATEGY_ID = "reciprocal-rank-fusion.v1"


@dataclass(frozen=True, slots=True)
class RankFusionConfiguration:
    """Typed RRF parameters."""

    rrf_k: int = 60

    def __post_init__(self) -> None:
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise RankFusionContractError("rrf_k must be a positive int")


@dataclass(frozen=True, slots=True)
class RankedRetrievalCandidate(Generic[TPayload]):
    candidate_id: str
    rank: int
    payload: TPayload | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise RankFusionContractError("candidate_id must be a non-empty str")
        if type(self.rank) is not int or self.rank < 0:
            raise RankFusionContractError("rank must be a non-negative int")


@dataclass(frozen=True, slots=True)
class RankedRetrievalChannel(Generic[TPayload]):
    channel_key: str
    candidates: tuple[RankedRetrievalCandidate[TPayload], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.channel_key, str) or not self.channel_key.strip():
            raise RankFusionContractError("channel_key must be a non-empty str")
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")


@dataclass(frozen=True, slots=True)
class RankFusionChannelEvidence:
    channel_key: str
    channel_rank: int
    reciprocal_rank_contribution: float

    def __post_init__(self) -> None:
        if not isinstance(self.channel_key, str) or not self.channel_key.strip():
            raise RankFusionContractError("channel_key must be a non-empty str")
        if type(self.channel_rank) is not int or self.channel_rank < 0:
            raise RankFusionContractError("channel_rank must be a non-negative int")
        _validate_positive_finite(
            self.reciprocal_rank_contribution,
            field_name="RankFusionChannelEvidence.reciprocal_rank_contribution",
        )


@dataclass(frozen=True, slots=True)
class FusedRankedCandidate(Generic[TPayload]):
    candidate_id: str
    fused_rank: int
    fusion_score: float
    supporting_channel_count: int
    evidence: tuple[RankFusionChannelEvidence, ...]
    payload: TPayload | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id.strip():
            raise RankFusionContractError("candidate_id must be a non-empty str")
        if type(self.fused_rank) is not int or self.fused_rank < 0:
            raise RankFusionContractError("fused_rank must be a non-negative int")
        _validate_positive_finite(self.fusion_score, field_name="FusedRankedCandidate.fusion_score")
        if type(self.supporting_channel_count) is not int or self.supporting_channel_count < 1:
            raise RankFusionContractError("supporting_channel_count must be a positive int")
        if not isinstance(self.evidence, tuple):
            raise TypeError("evidence must be a tuple")
        if len(self.evidence) != self.supporting_channel_count:
            raise RankFusionContractError(
                "supporting_channel_count must match evidence length"
            )


@dataclass(frozen=True, slots=True)
class RankFusionResult(Generic[TPayload]):
    candidates: tuple[FusedRankedCandidate[TPayload], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.candidates, tuple):
            raise TypeError("candidates must be a tuple")
        for rank, candidate in enumerate(self.candidates):
            if candidate.fused_rank != rank:
                raise RankFusionContractError("fused ranks must be contiguous from zero")


class RankFusionStrategyPort(Protocol[TPayload]):
    """Replaceable rank fusion strategy (RRF or alternatives)."""

    @property
    def strategy_id(self) -> str:
        ...

    def fuse(
        self,
        channels: tuple[RankedRetrievalChannel[TPayload], ...],
        *,
        limit: int | None = None,
    ) -> RankFusionResult[TPayload]:
        ...


def _validate_positive_finite(value: float, *, field_name: str) -> None:
    if type(value) not in (int, float):
        raise RankFusionContractError(f"{field_name} must be a float or int")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise RankFusionContractError(f"{field_name} must be a finite value greater than zero")
