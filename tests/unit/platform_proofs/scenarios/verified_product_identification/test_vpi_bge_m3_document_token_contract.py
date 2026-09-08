"""Contract tests for canonical 768 / effective 770 BGE-M3 document token semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_document_embedding_input_policy import (
    TokenBudgetDocumentEmbeddingInputPolicy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_truncation import (
    truncate_to_token_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING,
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.qualification.bounded_cuda_batch.metrics import (
    verify_effective_provider_token_ceiling,
)

pytestmark = pytest.mark.unit

_SPECIAL_TOKEN_CLS = 0
_SPECIAL_TOKEN_SEP = 1


@dataclass(frozen=True, slots=True)
class BgeM3RoundTripFakeCodec:
    """Deterministic codec simulating BGE-M3 truncation vs provider special tokens."""

    def encode_text_tokens(self, text: str) -> tuple[int, ...]:
        return tuple(ord(char) for char in text)

    def encode_provider_text_tokens(self, text: str) -> tuple[int, ...]:
        content = self.encode_text_tokens(text)
        return (_SPECIAL_TOKEN_CLS, *content, _SPECIAL_TOKEN_SEP)

    def decode_text_tokens(self, token_ids: Sequence[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def _canonical_policy(
    codec: BgeM3RoundTripFakeCodec,
    *,
    token_budget: int = VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
) -> TokenBudgetDocumentEmbeddingInputPolicy:
    return TokenBudgetDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=token_budget,
        tokenizer_codec=codec,
    )


def test_canonical_truncation_budget_is_768() -> None:
    assert VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET == 768


def test_effective_provider_token_ceiling_is_770() -> None:
    assert VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING == 770


def test_policy_version_unchanged() -> None:
    assert VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION == (
        "vpi-bge-m3-document-token-budget-768-v1"
    )


def test_text_below_truncation_budget_remains_unchanged() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec)
    short_text = "within budget"
    assert policy.apply_document(short_text) == short_text
    assert len(codec.encode_provider_text_tokens(short_text)) <= (
        VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING
    )


def test_truncation_is_deterministic_via_canonical_tokenizer_path() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec, token_budget=20)
    source_text = "abcdefghijklmnopqrstuvwxyz"
    first = policy.apply_document(source_text)
    second = policy.apply_document(source_text)
    assert first == second
    assert len(codec.encode_text_tokens(first)) <= 20


def test_truncated_output_retokenizes_within_effective_provider_ceiling() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec)
    source_text = "x" * (VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET + 50)
    bounded = policy.apply_document(source_text)
    provider_token_count = len(codec.encode_provider_text_tokens(bounded))
    assert provider_token_count <= VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING
    verify_effective_provider_token_ceiling((provider_token_count,))


def test_boundary_truncation_may_exceed_truncation_budget_after_provider_special_tokens() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec)
    source_text = "x" * VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET
    bounded = policy.apply_document(source_text)
    truncation_tokens = len(codec.encode_text_tokens(bounded))
    provider_tokens = len(codec.encode_provider_text_tokens(bounded))
    assert truncation_tokens == VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET
    assert provider_tokens > VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET
    assert provider_tokens == VPI_BGE_M3_DOCUMENT_EFFECTIVE_TOKEN_CEILING


def test_unicode_round_trip_remains_deterministic() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec, token_budget=5)
    text = "ąćęłńóśźż"
    first = policy.apply_document(text)
    second = policy.apply_document(text)
    assert first == second
    assert isinstance(first, str)


def test_empty_input_remains_deterministic() -> None:
    codec = BgeM3RoundTripFakeCodec()
    policy = _canonical_policy(codec)
    assert policy.apply_document("") == ""


def test_truncate_to_token_limit_matches_canonical_path() -> None:
    codec = BgeM3RoundTripFakeCodec()
    text = "abcdefghijklmnopqrstuvwxyz"
    bounded = truncate_to_token_limit(
        text,
        token_limit=VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
        encode=codec.encode_text_tokens,
        decode=codec.decode_text_tokens,
    )
    assert len(codec.encode_text_tokens(bounded)) <= VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET


def test_schema_versions_unchanged() -> None:
    assert DATA_PACK_VERSION
    assert RELATIONAL_SCHEMA_VERSION
    assert EMBEDDING_SCHEMA_VERSION


def test_content_identity_semantics_unchanged() -> None:
    source_dataset = SourceDatasetIdentity(
        dataset_name="offers",
        dataset_path="selected_offers.parquet",
        dataset_sha256="a" * 64,
        dataset_record_count=50,
    )
    identity = EmbeddingPackIdentity(
        provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        model=VPI_CANONICAL_EMBEDDING_MODEL,
        model_revision=VPI_CANONICAL_EMBEDDING_REVISION,
        artifact_fingerprint=None,
        dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        embedding_configuration_version="v1",
        input_policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    )
    content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    assert content_identity
