"""Typed JSON document models for semantic representation evidence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.contracts import (
    RepresentationReductionMetrics,
    SemanticRepresentationPolicy,
    SemanticRepresentationResult,
    TruncationStrategy,
)


@dataclass(frozen=True, slots=True)
class SemanticRepresentationPolicyJsonDocument:
    max_characters: int | None
    max_tokens: int | None
    preserved_fields: tuple[str, ...]
    truncation_strategy: str


@dataclass(frozen=True, slots=True)
class SemanticRepresentationResultJsonDocument:
    source_ref_offer_id: str
    source_ref_catalog_id: str
    source_ref_revision: str | None
    original_character_count: int
    final_character_count: int
    estimated_tokens: int
    truncated: bool
    preserved_field_count: int
    representation_text: str


@dataclass(frozen=True, slots=True)
class RepresentationReductionMetricsJsonDocument:
    before_chars: int
    after_chars: int
    reduction_ratio: float
    before_tokens: int
    after_tokens: int


def build_policy_json_document(
    policy: SemanticRepresentationPolicy,
) -> SemanticRepresentationPolicyJsonDocument:
    return SemanticRepresentationPolicyJsonDocument(
        max_characters=policy.max_characters,
        max_tokens=policy.max_tokens,
        preserved_fields=policy.preserved_fields,
        truncation_strategy=policy.truncation_strategy.value,
    )


def build_result_json_document(
    result: SemanticRepresentationResult,
) -> SemanticRepresentationResultJsonDocument:
    return SemanticRepresentationResultJsonDocument(
        source_ref_offer_id=result.source_ref.offer_id.value,
        source_ref_catalog_id=result.source_ref.catalog_id,
        source_ref_revision=result.source_ref.source_revision,
        original_character_count=result.original_character_count,
        final_character_count=result.final_character_count,
        estimated_tokens=result.estimated_tokens,
        truncated=result.truncated,
        preserved_field_count=result.preserved_field_count,
        representation_text=result.representation_text,
    )


def build_reduction_metrics_json_document(
    metrics: RepresentationReductionMetrics,
) -> RepresentationReductionMetricsJsonDocument:
    return RepresentationReductionMetricsJsonDocument(
        before_chars=metrics.before_chars,
        after_chars=metrics.after_chars,
        reduction_ratio=metrics.reduction_ratio,
        before_tokens=metrics.before_tokens,
        after_tokens=metrics.after_tokens,
    )


def serialize_policy_json(policy: SemanticRepresentationPolicy) -> str:
    document = build_policy_json_document(policy)
    return json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)


def serialize_result_json(result: SemanticRepresentationResult) -> str:
    document = build_result_json_document(result)
    return json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)


def serialize_reduction_metrics_json(metrics: RepresentationReductionMetrics) -> str:
    document = build_reduction_metrics_json_document(metrics)
    return json.dumps(asdict(document), ensure_ascii=False, sort_keys=True)


def parse_policy_json(payload: str) -> SemanticRepresentationPolicy:
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        msg = "policy JSON must be an object"
        raise ValueError(msg)

    max_characters = parsed.get("max_characters")
    max_tokens = parsed.get("max_tokens")
    preserved_fields_raw = parsed.get("preserved_fields")
    truncation_strategy_raw = parsed.get("truncation_strategy")

    if max_characters is not None and not isinstance(max_characters, int):
        msg = "max_characters must be an integer or null"
        raise ValueError(msg)
    if max_tokens is not None and not isinstance(max_tokens, int):
        msg = "max_tokens must be an integer or null"
        raise ValueError(msg)
    if not isinstance(preserved_fields_raw, list) or not all(
        isinstance(item, str) for item in preserved_fields_raw
    ):
        msg = "preserved_fields must be a list of strings"
        raise ValueError(msg)
    if not isinstance(truncation_strategy_raw, str):
        msg = "truncation_strategy must be a string"
        raise ValueError(msg)

    return SemanticRepresentationPolicy(
        max_characters=max_characters,
        max_tokens=max_tokens,
        preserved_fields=tuple(preserved_fields_raw),
        truncation_strategy=TruncationStrategy(truncation_strategy_raw),
    )


def parse_reduction_metrics_json(payload: str) -> RepresentationReductionMetrics:
    parsed = json.loads(payload)
    if not isinstance(parsed, dict):
        msg = "reduction metrics JSON must be an object"
        raise ValueError(msg)

    required_int_fields = ("before_chars", "after_chars", "before_tokens", "after_tokens")
    for field_name in required_int_fields:
        value = parsed.get(field_name)
        if not isinstance(value, int):
            msg = f"{field_name} must be an integer"
            raise ValueError(msg)

    reduction_ratio = parsed.get("reduction_ratio")
    if not isinstance(reduction_ratio, (int, float)):
        msg = "reduction_ratio must be numeric"
        raise ValueError(msg)

    return RepresentationReductionMetrics(
        before_chars=parsed["before_chars"],
        after_chars=parsed["after_chars"],
        reduction_ratio=float(reduction_ratio),
        before_tokens=parsed["before_tokens"],
        after_tokens=parsed["after_tokens"],
    )
