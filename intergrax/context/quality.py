# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Context quality scoring utilities (Phase CE-1.5)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ContextChunkSignal(BaseModel):
    chunk_id: str
    content_hash: str
    relevance_score: float
    freshness_score: float
    confidence_score: float

    @field_validator("relevance_score", "freshness_score", "confidence_score")
    @classmethod
    def _validate_ratio(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("Context quality scores must be in range [0.0, 1.0]")
        return value


class ContextQualityThresholds(BaseModel):
    min_relevance: float = 0.60
    min_freshness: float = 0.50
    min_confidence: float = 0.70
    min_composite_score: float = 0.65


class ContextChunkQualityRecord(BaseModel):
    chunk_id: str
    composite_score: float
    passed: bool
    reasons: list[str] = Field(default_factory=list)


class ContextEngineeringReport(BaseModel):
    schema_version: str = "1.0.0"
    thresholds: ContextQualityThresholds = Field(default_factory=ContextQualityThresholds)
    records: list[ContextChunkQualityRecord] = Field(default_factory=list)
    deduplicated_chunk_ids: list[str] = Field(default_factory=list)
    suppressed_duplicate_ids: list[str] = Field(default_factory=list)


def evaluate_context_engineering(
    *,
    chunks: list[ContextChunkSignal],
    thresholds: ContextQualityThresholds | None = None,
) -> ContextEngineeringReport:
    policy = thresholds or ContextQualityThresholds()
    deduplicated, suppressed = deduplicate_context_chunks(chunks)
    records: list[ContextChunkQualityRecord] = []
    for chunk in deduplicated:
        composite = _composite_score(chunk)
        reasons: list[str] = []
        if chunk.relevance_score < policy.min_relevance:
            reasons.append("Relevance score below threshold")
        if chunk.freshness_score < policy.min_freshness:
            reasons.append("Freshness score below threshold")
        if chunk.confidence_score < policy.min_confidence:
            reasons.append("Confidence score below threshold")
        if composite < policy.min_composite_score:
            reasons.append("Composite context quality score below threshold")
        records.append(
            ContextChunkQualityRecord(
                chunk_id=chunk.chunk_id,
                composite_score=composite,
                passed=not reasons,
                reasons=reasons,
            )
        )
    return ContextEngineeringReport(
        thresholds=policy,
        records=records,
        deduplicated_chunk_ids=[chunk.chunk_id for chunk in deduplicated],
        suppressed_duplicate_ids=suppressed,
    )


def deduplicate_context_chunks(
    chunks: list[ContextChunkSignal],
) -> tuple[list[ContextChunkSignal], list[str]]:
    unique_chunks: list[ContextChunkSignal] = []
    suppressed_ids: list[str] = []
    seen_hashes: set[str] = set()
    for chunk in chunks:
        if chunk.content_hash in seen_hashes:
            suppressed_ids.append(chunk.chunk_id)
            continue
        seen_hashes.add(chunk.content_hash)
        unique_chunks.append(chunk)
    return unique_chunks, suppressed_ids


def _composite_score(chunk: ContextChunkSignal) -> float:
    return (
        chunk.relevance_score * 0.40
        + chunk.freshness_score * 0.25
        + chunk.confidence_score * 0.35
    )
