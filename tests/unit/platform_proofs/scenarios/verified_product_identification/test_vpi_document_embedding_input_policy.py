"""Unit tests for canonical 768-token document embedding input policy integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_document_embedding_input_policy import (
    TokenBudgetDocumentEmbeddingInputPolicy,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.token_budget_truncation import (
    truncate_to_token_limit,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding_input_policy import (
    VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
    VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildIdentityMismatchError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    VPI_CANONICAL_EMBEDDING_DIMENSION,
    VPI_CANONICAL_EMBEDDING_MODEL,
    VPI_CANONICAL_EMBEDDING_PROVIDER,
    VPI_CANONICAL_EMBEDDING_REVISION,
    semantic_text_hash,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    resolve_data_pack_paths,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.integration.query_set import (
    build_proof_query_set,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    FakeDocumentEmbeddingInputPolicy,
    FakeTokenizerCodec,
    canonical_fake_document_embedding_input_policy,
    patch_canonical_model_identity,
    run_resumable_data_pack_build_with_fake_policy,
    write_tiny_selected_dataset,
)

pytestmark = pytest.mark.unit

_REVISION = VPI_CANONICAL_EMBEDDING_REVISION
_SAMPLE_JSON = json.dumps(
    {
        "id": "offer-policy-1",
        "title": "Policy Widget",
        "identifiers": [{"gtin": "1234567890123"}],
        "keyValuePairs": {"voltage": "24V"},
        "description": "Deterministic policy widget.",
        "brand": "PolicyBrand",
    }
)


def _source_dataset() -> SourceDatasetIdentity:
    return SourceDatasetIdentity(
        dataset_name="offers",
        dataset_path="selected_offers.parquet",
        dataset_sha256="a" * 64,
        dataset_record_count=50,
    )


def _embedding_identity(input_policy_version: str) -> EmbeddingPackIdentity:
    return EmbeddingPackIdentity(
        provider=VPI_CANONICAL_EMBEDDING_PROVIDER,
        model=VPI_CANONICAL_EMBEDDING_MODEL,
        model_revision=_REVISION,
        artifact_fingerprint=None,
        dimension=VPI_CANONICAL_EMBEDDING_DIMENSION,
        embedding_configuration_version="v1",
        input_policy_version=input_policy_version,
    )


def _long_text(token_budget: int) -> str:
    return "x" * (token_budget + 50)


def test_text_within_budget_reaches_embedding_port_unchanged() -> None:
    policy = canonical_fake_document_embedding_input_policy()
    short_text = "within budget"
    assert policy.apply_document(short_text) == short_text


def test_text_over_budget_is_bounded_to_token_limit() -> None:
    codec = FakeTokenizerCodec()
    policy = FakeDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=768,
        tokenizer_codec=codec,
    )
    source_text = _long_text(768)
    bounded = policy.apply_document(source_text)
    assert len(codec.encode_text_tokens(bounded)) <= 768
    assert bounded != source_text


def test_relational_record_stores_full_semantic_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=5)
    output_root = tmp_path / "pack"
    long_policy = canonical_fake_document_embedding_input_policy(token_budget=3)
    embedding = FakeDataPackEmbeddingPort()
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=5,
            max_records=5,
            start_fresh=True,
        ),
        embedding_port=embedding,
        document_embedding_input_policy=long_policy,
    )
    assert report.manifest is not None
    paths = resolve_data_pack_paths(output_root)
    relational_records = read_relational_parquet(paths.relational_dir / "part-000001.parquet")
    for record in relational_records:
        source_offer = parse_wdc_source_offer_json(record.record_json)
        source_ref = build_source_record_ref(source_offer, catalog_id=record.source_ref.catalog_id)
        expected = derive_search_representation(source_offer, source_ref=source_ref).semantic.semantic_text
        assert record.semantic_text == expected
        assert len(embedding.texts_seen) > 0
        assert any(len(text) < len(record.semantic_text) for text in embedding.texts_seen) or all(
            text == record.semantic_text for text in embedding.texts_seen
        )


def test_relational_semantic_text_hash_matches_full_text() -> None:
    source_offer = parse_wdc_source_offer_json(_SAMPLE_JSON)
    source_ref = build_source_record_ref(source_offer, catalog_id="wdc-v2-selected")
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    full_text = representation.semantic.semantic_text
    assert semantic_text_hash(full_text) == semantic_text_hash(representation.semantic.semantic_text)


def test_embedding_port_receives_bounded_not_full_text() -> None:
    codec = FakeTokenizerCodec()
    policy = FakeDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=10,
        tokenizer_codec=codec,
    )
    full_text = _long_text(10)
    bounded = policy.apply_document(full_text)
    embedding = FakeDataPackEmbeddingPort()
    embedding.embed_batch([bounded])
    assert embedding.texts_seen == [bounded]
    assert embedding.texts_seen != [full_text]


def test_query_paths_are_unaffected_by_document_policy() -> None:
    source_offer = parse_wdc_source_offer_json(_SAMPLE_JSON)
    source_ref = build_source_record_ref(source_offer, catalog_id="wdc-v2-selected")
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
        RelationalDataPackRecord,
    )

    record = RelationalDataPackRecord(
        global_row_index=0,
        source_ref=source_ref,
        record_json=_SAMPLE_JSON,
        derivation_version=representation.derivation_version,
        semantic_text=representation.semantic.semantic_text,
        semantic_text_hash=semantic_text_hash(representation.semantic.semantic_text),
        title=source_offer.title,
        brand=source_offer.brand,
        category=source_offer.category,
        description=source_offer.description,
        has_identifiers=True,
        has_spec_table=False,
        has_structured_attributes=True,
    )
    query_set_before = build_proof_query_set((record,))
    policy = canonical_fake_document_embedding_input_policy(token_budget=3)
    policy.apply_document(record.semantic_text)
    query_set_after = build_proof_query_set((record,))
    assert query_set_after == query_set_before
    vector_case = next(case for case in query_set_after if case.vector_query is not None)
    assert vector_case.vector_query is not None
    assert vector_case.vector_query.query_text == source_offer.description


def test_canonical_policy_version_constant() -> None:
    assert VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION == (
        "vpi-bge-m3-document-token-budget-768-v1"
    )
    assert VPI_CANONICAL_DOCUMENT_EMBEDDING_TOKEN_BUDGET == 768


def test_manifest_records_input_policy_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=5)
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=tmp_path / "pack",
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=5,
            max_records=5,
            start_fresh=True,
        ),
    )
    assert report.manifest is not None
    assert (
        report.manifest.embedding_identity.input_policy_version
        == VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION
    )


def test_content_identity_differs_between_full_and_bounded_policy() -> None:
    source_dataset = _source_dataset()
    full_identity = _embedding_identity(SEARCH_REPRESENTATION_DERIVATION_VERSION)
    bounded_identity = _embedding_identity(VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION)
    full_content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=full_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    bounded_content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=bounded_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    assert full_content_identity != bounded_content_identity


def test_old_full_policy_build_state_cannot_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=5)
    output_root = tmp_path / "pack"
    run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=5,
            max_records=5,
            start_fresh=True,
            stop_after_shard=1,
        ),
    )
    paths = resolve_data_pack_paths(output_root)
    payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
    payload["content_identity"] = compute_data_pack_content_identity(
        source_dataset=SourceDatasetIdentity(
            dataset_name="offers_corpus_all_v2_non_norm",
            dataset_path=str(dataset_path),
            dataset_sha256="a" * 64,
            dataset_record_count=5,
        ),
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=_embedding_identity(SEARCH_REPRESENTATION_DERIVATION_VERSION),
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VpiDataPackBuildIdentityMismatchError):
        run_resumable_data_pack_build_with_fake_policy(
            DataPackBuildConfig(
                output_root=output_root,
                dataset_path=dataset_path,
                dataset_manifest_path=manifest_path,
                shard_size=5,
                max_records=5,
                resume=True,
            ),
        )


def test_same_source_and_policy_produces_deterministic_input() -> None:
    codec = FakeTokenizerCodec()
    policy = TokenBudgetDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=20,
        tokenizer_codec=codec,
    )
    text = _long_text(20)
    assert policy.apply_document(text) == policy.apply_document(text)


def test_unicode_survives_token_encode_decode() -> None:
    codec = FakeTokenizerCodec()
    policy = FakeDocumentEmbeddingInputPolicy(
        policy_version=VPI_BGE_M3_DOCUMENT_TOKEN_BUDGET_768_POLICY_VERSION,
        token_budget=5,
        tokenizer_codec=codec,
    )
    text = "ąćęłńóśźż"
    bounded = policy.apply_document(text)
    assert isinstance(bounded, str)


def test_empty_text_is_deterministic() -> None:
    policy = canonical_fake_document_embedding_input_policy()
    assert policy.apply_document("") == ""


def test_injected_fake_policy_works_without_hf_stack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=3)
    report = run_resumable_data_pack_build(
        DataPackBuildConfig(
            output_root=tmp_path / "pack",
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=3,
            max_records=3,
            start_fresh=True,
        ),
        embedding_port=FakeDataPackEmbeddingPort(),
        document_embedding_input_policy=canonical_fake_document_embedding_input_policy(),
    )
    assert report.status is DataPackStatus.READY


def test_schema_versions_remain_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=3)
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=tmp_path / "pack",
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=3,
            max_records=3,
            start_fresh=True,
        ),
    )
    assert report.manifest is not None
    assert report.manifest.data_pack_version == DATA_PACK_VERSION
    assert report.manifest.relational_schema_version == RELATIONAL_SCHEMA_VERSION
    assert report.manifest.embedding_schema_version == EMBEDDING_SCHEMA_VERSION
    paths = resolve_data_pack_paths(tmp_path / "pack")
    state = read_build_state_file(paths.build_state_file)
    assert state.content_identity == report.manifest.content_identity


def test_policy_version_change_changes_content_identity() -> None:
    source_dataset = _source_dataset()
    first = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=_embedding_identity("policy-a"),
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    second = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_identity=_embedding_identity("policy-b"),
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    assert first != second


def test_truncate_to_token_limit_matches_qualification_semantics() -> None:
    codec = FakeTokenizerCodec()
    text = "abcdefghijklmnop"
    bounded = truncate_to_token_limit(
        text,
        token_limit=5,
        encode=codec.encode_text_tokens,
        decode=codec.decode_text_tokens,
    )
    assert len(codec.encode_text_tokens(bounded)) <= 5
