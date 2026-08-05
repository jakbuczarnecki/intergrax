# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Any, Optional, Sequence

from intergrax.knowledge.contracts import KnowledgeDocument


class RerankerNormalizationMode(str, Enum):
    MINMAX = "minmax"
    ZSCORE = "zscore"


class RerankerField(str, Enum):
    CONTENT = "content"
    METADATA = "metadata"

    ORIGINAL_SCORE = "similarity_score"

    RERANK_SCORE = "rerank_score"
    FUSION_SCORE = "fusion_score"
    RERANK_RANK = "rank"


def _finite_score(value: object, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be a finite number, not bool")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(converted):
        raise ValueError(f"{field_name} must be finite")
    return converted


def _rank(value: object, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise TypeError(f"{field_name} must be an exact non-negative int")
    return value


def validate_limit(limit: int | None) -> int | None:
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be an exact positive int or None")
    return limit


def validate_candidates(
    candidates: Sequence["RerankerCandidate"],
) -> tuple["RerankerCandidate", ...]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be a sequence of RerankerCandidate")
    normalized = tuple(candidates)
    for candidate in normalized:
        if not isinstance(candidate, RerankerCandidate):
            raise TypeError("candidates must contain only RerankerCandidate values")
    return normalized


@dataclass(frozen=True, slots=True)
class RerankerCandidate:
    """Immutable native reranker input.

    ``original_score`` is the normalized retrieval similarity in ``[0, 1]``.
    All document identity, scope, provenance and user metadata remain on the
    ``KnowledgeDocument``; they are not copied into parallel candidate fields.
    """

    document: KnowledgeDocument
    original_score: float
    original_rank: int
    channel: str
    vector_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document, KnowledgeDocument):
            raise TypeError("document must be a KnowledgeDocument")
        document = KnowledgeDocument.model_validate(
            self.document.model_dump(mode="python")
        )
        score = _finite_score(self.original_score, field_name="original_score")
        if not 0.0 <= score <= 1.0:
            raise ValueError("original_score must be in the inclusive range [0, 1]")
        rank = _rank(self.original_rank, field_name="original_rank")
        if not isinstance(self.channel, str) or not self.channel.strip():
            raise ValueError("channel must be a non-empty string")
        vector_id = self.vector_id
        if vector_id is not None and (
            not isinstance(vector_id, str) or not vector_id.strip()
        ):
            raise ValueError("vector_id must be a non-empty string when provided")
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "original_score", score)
        object.__setattr__(self, "original_rank", rank)
        object.__setattr__(self, "vector_id", vector_id)

    @classmethod
    def from_retrieval_hit(cls, hit: object) -> "RerankerCandidate":
        from intergrax.rag.retrievers.contracts.base_retriever import RetrievalHit

        if not isinstance(hit, RetrievalHit):
            raise TypeError("hit must be a RetrievalHit")
        return cls(
            document=hit.document,
            original_score=hit.score,
            original_rank=hit.rank,
            channel=hit.channel,
            vector_id=hit.vector_id,
        )

    @property
    def identity_key(self) -> tuple[str, str | None, str, str | None]:
        return (
            self.document.scope.tenant_id,
            self.document.scope.namespace,
            self.document.identity.document_id,
            self.vector_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.model_dump(mode="json"),
            "original_score": self.original_score,
            "original_rank": self.original_rank,
            "channel": self.channel,
            "vector_id": self.vector_id,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class RerankerResult:
    """Immutable reranker output.

    Reranker scores are provider/final scores and must be finite. Their range
    is provider-defined; normalization and fusion may constrain them to [0, 1].
    ``rank`` is the final zero-based stable rank.
    """

    candidate: RerankerCandidate
    rerank_score: float
    rank: int
    fusion_score: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RerankerCandidate):
            raise TypeError("candidate must be a RerankerCandidate")
        rerank_score = _finite_score(self.rerank_score, field_name="rerank_score")
        fusion_score = self.fusion_score
        if fusion_score is not None:
            fusion_score = _finite_score(fusion_score, field_name="fusion_score")
        rank = _rank(self.rank, field_name="rank")
        object.__setattr__(self, "rerank_score", rerank_score)
        object.__setattr__(self, "fusion_score", fusion_score)
        object.__setattr__(self, "rank", rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "rerank_score": self.rerank_score,
            "rank": self.rank,
            "fusion_score": self.fusion_score,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )