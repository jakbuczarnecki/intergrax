"""Unit tests for VPI universal data pack contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.validation import (
    assert_validation_pass,
    validate_cross_artifact_identity,
    validate_embedding_records,
    validate_relational_records,
    validate_semantic_text_hashes,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    semantic_text_hash,
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    EmbeddingPackIdentity,
    SampleIdentity,
    SourceDatasetIdentity,
    manifest_from_json_dict,
    manifest_to_json_dict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)

pytestmark = pytest.mark.unit

_SAMPLE_JSON = json.dumps(
    {
        "id": "offer-proof-1",
        "cluster_id": 1,
        "category": "Electronics",
        "identifiers": [{"key": "gtin", "value": "1234567890123"}],
        "title": "Proof Widget",
        "description": "A deterministic proof widget for VPI data pack tests.",
        "brand": "ProofBrand",
        "price": "9.99",
        "keyValuePairs": {"Voltage": "24V"},
        "specTableContent": "| Voltage | 24V |",
    }
)


def _relational_record() -> RelationalDataPackRecord:
    source_offer = parse_wdc_source_offer_json(_SAMPLE_JSON)
    source_ref = build_source_record_ref(source_offer, catalog_id="wdc-v2-selected")
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    semantic_text = representation.semantic.semantic_text
    return RelationalDataPackRecord(
        global_row_index=0,
        source_ref=source_ref,
        record_json=_SAMPLE_JSON,
        derivation_version=representation.derivation_version,
        semantic_text=semantic_text,
        semantic_text_hash=semantic_text_hash(semantic_text),
        title=source_offer.title,
        brand=source_offer.brand,
        category=source_offer.category,
        description=source_offer.description,
        has_identifiers=True,
        has_spec_table=True,
        has_structured_attributes=True,
    )


def _embedding_record(relational: RelationalDataPackRecord) -> EmbeddingDataPackRecord:
    vector = tuple(0.01 * index for index in range(1024))
    return EmbeddingDataPackRecord(
        logical_point_id=search_representation_point_id(
            catalog_id=relational.source_ref.catalog_id,
            offer_id=relational.source_ref.offer_id.value,
            derivation_version=relational.derivation_version,
        ),
        source_ref=relational.source_ref,
        derivation_version=relational.derivation_version,
        semantic_text_hash=relational.semantic_text_hash,
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_model_revision=None,
        embedding_dimension=1024,
        dense_embedding=vector,
    )


def test_manifest_round_trip() -> None:
    manifest = DataPackManifest(
        data_pack_version="vpi.data_pack/1.0.0",
        scenario_id="verified_product_identification",
        source_dataset=SourceDatasetIdentity(
            dataset_name="offers",
            dataset_path="/tmp/selected_offers.parquet",
            dataset_sha256="abc",
            dataset_record_count=50,
        ),
        source_record_count=50,
        sample_identity=SampleIdentity(
            sample_version="proof-50/1.0.0",
            sample_seed=42,
            selected_record_refs=("wdc-v2-selected:offer-1",),
        ),
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=EmbeddingPackIdentity(
            provider="hf",
            model="BAAI/bge-m3",
            model_revision=None,
            dimension=1024,
            embedding_configuration_version="v1",
            input_policy_version="v2",
            execution_configuration_identity="device=cuda",
        ),
        relational_format="parquet/v1",
        embedding_format="parquet/v1",
        shard_count=1,
        record_count=50,
        created_at_utc="2026-09-06T00:00:00+00:00",
        status=DataPackStatus.READY,
        checksums_path="checksums/SHA256SUMS",
        shards_index_path="indexes/shards.json",
        relational_shard_file="part-000001.parquet",
        embedding_shard_file="part-000001.parquet",
    )
    restored = manifest_from_json_dict(manifest_to_json_dict(manifest))
    assert restored == manifest


def test_parquet_round_trip_and_validation(tmp_path: Path) -> None:
    relational = _relational_record()
    embedding = _embedding_record(relational)
    relational_path = tmp_path / "relational.parquet"
    embedding_path = tmp_path / "embedding.parquet"
    write_relational_parquet(relational_path, (relational,))
    write_embedding_parquet(embedding_path, (embedding,), embedding_dimension=1024)
    restored_relational = read_relational_parquet(relational_path)
    restored_embedding = read_embedding_parquet(embedding_path, expected_dimension=1024)
    assert restored_relational[0].source_ref == relational.source_ref
    assert len(restored_embedding[0].dense_embedding) == len(embedding.dense_embedding)
    assert_validation_pass(
        validate_relational_records(restored_relational, expected_count=1),
        stage="relational",
    )
    assert_validation_pass(
        validate_embedding_records(
            restored_embedding,
            expected_count=1,
            expected_dimension=1024,
        ),
        stage="embedding",
    )
    assert_validation_pass(
        validate_cross_artifact_identity(restored_relational, restored_embedding),
        stage="cross_ref",
    )
    assert_validation_pass(
        validate_semantic_text_hashes(restored_relational, restored_embedding),
        stage="semantic_hash",
    )


def test_checksum_verification(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"proof-50")
    checksums = tmp_path / "SHA256SUMS"
    write_sha256sums(checksums, (("artifact.bin", target),))
    verify_sha256sums(checksums, tmp_path)
    target.write_bytes(b"tampered")
    with pytest.raises(Exception):
        verify_sha256sums(checksums, tmp_path)


def test_duplicate_source_ref_rejected() -> None:
    relational = _relational_record()
    duplicate = RelationalDataPackRecord(
        global_row_index=1,
        source_ref=relational.source_ref,
        record_json=relational.record_json,
        derivation_version=relational.derivation_version,
        semantic_text=relational.semantic_text,
        semantic_text_hash=relational.semantic_text_hash,
        title=relational.title,
        brand=relational.brand,
        category=relational.category,
        description=relational.description,
        has_identifiers=relational.has_identifiers,
        has_spec_table=relational.has_spec_table,
        has_structured_attributes=relational.has_structured_attributes,
    )
    report = validate_relational_records((relational, duplicate), expected_count=2)
    assert report.status.value == "FAIL"


def test_data_pack_contracts_have_no_provider_imports() -> None:
    import ast

    contracts_root = (
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/contracts"
    )
    violations: list[str] = []
    for path in contracts_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and ".integrations." in node.module:
                violations.append(f"{path.name}:{node.module}")
    assert violations == []
