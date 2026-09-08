"""Unit tests for bounded tokenizer truncation qualification."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    WdcSourceOffer,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.classification import (
    QueryDifficultyClass,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.query_benchmark import (
    ArenaSourceRef,
    EmbeddingArenaQueryCase,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.results import (
    RetrievalQualityMetrics,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.stage_evaluation_scope import (
    EmbeddingArenaStageEvaluationScope,
    compute_stage_content_fingerprint,
)
from platform_proofs.scenarios.verified_product_identification.arena.contracts.versioning import (
    ARENA_QUERY_BENCHMARK_VERSION,
    ARENA_SAMPLE_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.arena.sampling.arena_sample import (
    ArenaSampleRecord,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.contracts import (
    RepresentationVariant,
    VariantQualityGateResult,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.evaluation import (
    compare_query_rankings,
    evaluate_quality_gate,
    select_winning_candidate,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_representation.truncation import (
    ProductRepresentationVariantPort,
    truncate_to_token_limit,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BOUNDED_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/qualification/bounded_representation"
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab = {char: index + 1 for index, char in enumerate("abcdefghijklmnopqrstuvwxyz ")}

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        if not text:
            return []
        return [self._vocab[char] for char in text.lower() if char in self._vocab]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        reverse = {value: key for key, value in self._vocab.items()}
        return "".join(reverse[token_id] for token_id in token_ids if token_id in reverse)


def _variant_port() -> ProductRepresentationVariantPort:
    tokenizer = _FakeTokenizer()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=True)

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    return ProductRepresentationVariantPort(
        encode=encode,
        decode=decode,
        count_tokens=lambda text: len(encode(text)),
    )


def _sample_offer(offer_id: str, title: str) -> WdcSourceOffer:
    return WdcSourceOffer(
        offer_id=offer_id,
        cluster_id=None,
        category="electronics",
        identifiers=(),
        title=title,
        description="Short",
        brand="Brand",
        price="1.00",
        key_value_pairs=(),
        spec_table_content=None,
    )


def test_full_variant_is_byte_for_byte_unchanged() -> None:
    port = _variant_port()
    text = "alpha beta gamma"
    assert port.apply(text, RepresentationVariant.FULL) == text


def test_bounded_output_never_exceeds_token_budget() -> None:
    port = _variant_port()
    text = "abcdefghijklmnopqrstuvwxyz"
    for variant in (
        RepresentationVariant.TOKEN_LIMIT_1024,
        RepresentationVariant.TOKEN_LIMIT_768,
        RepresentationVariant.TOKEN_LIMIT_512,
    ):
        limit = variant.token_limit()
        assert limit is not None
        bounded = port.apply(text, variant)
        assert port.count_tokens_for(bounded) <= limit


def test_repeated_transformation_is_identical() -> None:
    port = _variant_port()
    text = "abcdefghijklmnopqrstuvwxyz"
    variant = RepresentationVariant.TOKEN_LIMIT_512
    first = port.apply(text, variant)
    second = port.apply(text, variant)
    assert first == second


def test_empty_text_behavior_is_explicit() -> None:
    port = _variant_port()
    for variant in RepresentationVariant:
        assert port.apply("", variant) == ""


def test_unicode_text_remains_valid() -> None:
    port = _variant_port()
    text = "żółć éàü"
    bounded = port.apply(text, RepresentationVariant.TOKEN_LIMIT_512)
    assert isinstance(bounded, str)
    bounded.encode("utf-8")


def test_truncate_to_token_limit_helper() -> None:
    tokenizer = _FakeTokenizer()

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, add_special_tokens=True)

    def decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids, skip_special_tokens=False)

    text = "abcdefghijklmnopqrstuvwxyz"
    bounded = truncate_to_token_limit(text, token_limit=5, encode=encode, decode=decode)
    assert len(encode(bounded)) <= 5


def _scope_with_cases() -> EmbeddingArenaStageEvaluationScope:
    records = (
        ArenaSampleRecord(
            offer_id="offer-a",
            global_row_index=0,
            semantic_text="alpha",
            source_offer=_sample_offer("offer-a", "Alpha Product"),
            strata_tags=("has_brand",),
        ),
        ArenaSampleRecord(
            offer_id="offer-b",
            global_row_index=1,
            semantic_text="beta",
            source_offer=_sample_offer("offer-b", "Beta Product"),
            strata_tags=("has_brand",),
        ),
    )
    query_cases = (
        EmbeddingArenaQueryCase(
            case_id="q-0001",
            query_text="Alpha Product",
            difficulty=QueryDifficultyClass.TITLE_ONLY,
            relevant_source_refs=(ArenaSourceRef(offer_id="offer-a", global_row_index=0),),
            provenance="test",
            benchmark_only_cluster_evidence=None,
            hard_negative_offer_ids=("offer-b",),
            is_long_input_query=False,
        ),
    )
    offer_index = {record.offer_id: index for index, record in enumerate(records)}
    return EmbeddingArenaStageEvaluationScope(
        stage_name="test",
        records=records,
        query_cases=query_cases,
        offer_index=offer_index,
        corpus_size=len(records),
        benchmark_version=ARENA_QUERY_BENCHMARK_VERSION,
        sample_version=ARENA_SAMPLE_VERSION,
        content_fingerprint=compute_stage_content_fingerprint(records),
    )


def test_quality_gate_detects_recall_regression() -> None:
    control = RetrievalQualityMetrics(
        recall_at_1=1.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr_at_10=1.0,
        ndcg_at_10=1.0,
        query_count=1,
    )
    candidate = RetrievalQualityMetrics(
        recall_at_1=0.0,
        recall_at_5=1.0,
        recall_at_10=1.0,
        mrr_at_10=0.5,
        ndcg_at_10=0.7,
        query_count=1,
    )
    gate = evaluate_quality_gate(
        control,
        candidate,
        variant=RepresentationVariant.TOKEN_LIMIT_512,
        comparisons=(),
    )
    assert gate.passed is False
    assert gate.failure_reasons


def test_severe_regression_contract() -> None:
    scope = _scope_with_cases()
    comparisons = compare_query_rankings(
        scope,
        full_ranked=((0,),),
        candidate_ranked=((1,),),
    )
    assert comparisons[0].is_severe_regression is True
    assert comparisons[0].is_top1_regression is True


def test_select_winning_candidate_prefers_smallest_passing_budget() -> None:
    metrics = RetrievalQualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1)
    gates = {
        RepresentationVariant.FULL: VariantQualityGateResult(
            variant=RepresentationVariant.FULL,
            metrics=metrics,
            passed=True,
            failure_reasons=(),
        ),
        RepresentationVariant.TOKEN_LIMIT_512: VariantQualityGateResult(
            variant=RepresentationVariant.TOKEN_LIMIT_512,
            metrics=metrics,
            passed=True,
            failure_reasons=(),
        ),
        RepresentationVariant.TOKEN_LIMIT_768: VariantQualityGateResult(
            variant=RepresentationVariant.TOKEN_LIMIT_768,
            metrics=metrics,
            passed=True,
            failure_reasons=(),
        ),
        RepresentationVariant.TOKEN_LIMIT_1024: VariantQualityGateResult(
            variant=RepresentationVariant.TOKEN_LIMIT_1024,
            metrics=metrics,
            passed=True,
            failure_reasons=(),
        ),
    }
    assert select_winning_candidate(gates) is RepresentationVariant.TOKEN_LIMIT_512


def test_no_forbidden_contract_patterns() -> None:
    forbidden_fragments = (
        "dict[str, Any]",
        ": Any",
        "dict[str, object]",
        "getattr",
        "setattr",
        "hasattr",
    )
    for module_path in sorted(_BOUNDED_ROOT.glob("*.py")):
        source = module_path.read_text(encoding="utf-8")
        for fragment in forbidden_fragments:
            assert fragment not in source, f"{fragment} found in {module_path.name}"
