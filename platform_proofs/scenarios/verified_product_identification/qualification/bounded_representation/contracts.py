"""Immutable contracts for bounded tokenizer truncation qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)


class RepresentationVariant(StrEnum):
    """Product semantic text token budget — tokenizer truncation only."""

    FULL = "FULL"
    TOKEN_LIMIT_1024 = "TOKEN_LIMIT_1024"
    TOKEN_LIMIT_768 = "TOKEN_LIMIT_768"
    TOKEN_LIMIT_512 = "TOKEN_LIMIT_512"

    def token_limit(self) -> int | None:
        if self is RepresentationVariant.FULL:
            return None
        if self is RepresentationVariant.TOKEN_LIMIT_1024:
            return 1024
        if self is RepresentationVariant.TOKEN_LIMIT_768:
            return 768
        if self is RepresentationVariant.TOKEN_LIMIT_512:
            return 512
        msg = f"unsupported variant: {self}"
        raise ValueError(msg)


REPRESENTATION_VARIANT_ORDER: tuple[RepresentationVariant, ...] = (
    RepresentationVariant.FULL,
    RepresentationVariant.TOKEN_LIMIT_1024,
    RepresentationVariant.TOKEN_LIMIT_768,
    RepresentationVariant.TOKEN_LIMIT_512,
)


@dataclass(frozen=True, slots=True)
class PerQueryRetrievalComparison:
    query_id: str
    expected_offer_id: str
    full_rank: int | None
    candidate_rank: int | None
    rank_delta: int | None
    top1_changed: bool
    expected_lost_from_top5: bool
    expected_lost_from_top10: bool
    is_severe_regression: bool
    is_top1_regression: bool


@dataclass(frozen=True, slots=True)
class VariantQualityGateResult:
    variant: RepresentationVariant
    metrics: RetrievalQualityMetrics
    passed: bool
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BoundedRepresentationQualityReport:
    proof_identifier: str
    corpus_record_count: int
    query_count: int
    model_provider: str
    model_name: str
    model_revision: str
    model_dimension: int
    model_load_count: int
    control_metrics: RetrievalQualityMetrics
    variant_metrics: dict[str, RetrievalQualityMetrics]
    variant_gates: dict[str, VariantQualityGateResult]
    per_query_comparisons: dict[str, tuple[PerQueryRetrievalComparison, ...]]
    winning_candidate: str
    vector_quality_coverage_valid: bool
    status: str
