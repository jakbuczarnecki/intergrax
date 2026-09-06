"""Compatibility rejection and roundtrip tests for frozen VPI data pack v1."""

from __future__ import annotations

import json
from pathlib import Path

from dataclasses import replace

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    DataPackCompatibilityExpectations,
    default_v1_expectations,
    validate_data_pack_compatibility,
    validate_shard_index_contract,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackFormatError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
    manifest_from_json_dict,
    manifest_to_json_dict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
    ShardIndex,
    shard_index_from_json_dict,
    shard_index_to_json_dict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[4] / "fixtures"
_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def _source_dataset() -> SourceDatasetIdentity:
    return SourceDatasetIdentity(
        dataset_name="offers",
        dataset_path="processed/selected_offers.parquet",
        dataset_sha256="abc",
        dataset_record_count=50,
    )


def _embedding_identity(**overrides: object) -> EmbeddingPackIdentity:
    base = {
        "provider": "hf",
        "model": "BAAI/bge-m3",
        "model_revision": _REVISION,
        "artifact_fingerprint": None,
        "dimension": 1024,
        "embedding_configuration_version": "v1",
        "input_policy_version": "v2",
    }
    base.update(overrides)
    return EmbeddingPackIdentity(**base)


def _manifest(**overrides: object) -> DataPackManifest:
    mutable_overrides = dict(overrides)
    embedding_identity = mutable_overrides.pop("embedding_identity", _embedding_identity())
    relational_schema_version = mutable_overrides.pop(
        "relational_schema_version",
        RELATIONAL_SCHEMA_VERSION,
    )
    embedding_schema_version = mutable_overrides.pop(
        "embedding_schema_version",
        EMBEDDING_SCHEMA_VERSION,
    )
    source_dataset = mutable_overrides.pop("source_dataset", _source_dataset())
    content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=embedding_identity,
        relational_schema_version=relational_schema_version,
        embedding_schema_version=embedding_schema_version,
    )
    base = {
        "data_pack_version": DATA_PACK_VERSION,
        "content_identity": content_identity,
        "scenario_id": "verified_product_identification",
        "source_dataset": source_dataset,
        "source_record_count": 50,
        "sample_identity": None,
        "derivation_version": "v2",
        "semantic_text_version": "v2",
        "embedding_identity": embedding_identity,
        "relational_schema_version": relational_schema_version,
        "embedding_schema_version": embedding_schema_version,
        "relational_format": "parquet",
        "embedding_format": "parquet",
        "shard_count": 1,
        "record_count": 50,
        "created_at_utc": "2026-09-06T00:00:00+00:00",
        "status": DataPackStatus.READY,
        "checksums_path": "checksums/SHA256SUMS",
        "shards_index_path": "indexes/shards.json",
        "build_execution_provenance": None,
    }
    base.update(mutable_overrides)
    return DataPackManifest(**base)


def _expectations(**overrides: object) -> DataPackCompatibilityExpectations:
    base = default_v1_expectations(
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_model_revision=_REVISION,
        embedding_dimension=1024,
        source_dataset_sha256="abc",
    )
    return replace(base, **overrides)


def test_golden_manifest_fixture_parses() -> None:
    payload = json.loads((_FIXTURES / "vpi_data_pack_manifest_v1.json").read_text(encoding="utf-8"))
    manifest = manifest_from_json_dict(payload)
    assert manifest.data_pack_version == DATA_PACK_VERSION
    assert manifest.embedding_identity.model_revision == _REVISION


def test_manifest_round_trip_strict() -> None:
    manifest = _manifest()
    restored = manifest_from_json_dict(manifest_to_json_dict(manifest))
    assert restored == manifest


def test_shard_index_round_trip() -> None:
    shard_index = ShardIndex(
        shard_count=1,
        relational_shards=(
            ShardDescriptor(
                ordinal=1,
                relative_path="relational/part-000001.parquet",
                record_count=50,
                sha256="a" * 64,
                source_ref_count=50,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            ),
        ),
        embedding_shards=(
            ShardDescriptor(
                ordinal=1,
                relative_path="embeddings/part-000001.parquet",
                record_count=50,
                sha256="b" * 64,
                source_ref_count=50,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            ),
        ),
    )
    restored = shard_index_from_json_dict(shard_index_to_json_dict(shard_index))
    assert restored == shard_index


def test_strict_parser_rejects_wrong_type() -> None:
    payload = manifest_to_json_dict(_manifest())
    payload["record_count"] = "50"
    with pytest.raises(VpiDataPackFormatError):
        manifest_from_json_dict(payload)


def test_compatibility_rejects_wrong_revision() -> None:
    manifest = _manifest(
        embedding_identity=_embedding_identity(model_revision="deadbeefdeadbeefdeadbeefdeadbeefdeadbeef")
    )
    report = validate_data_pack_compatibility(manifest, expectations=_expectations())
    assert report.status.value == "FAIL"


def test_compatibility_rejects_wrong_schema_version() -> None:
    manifest = _manifest(relational_schema_version="vpi.relational/2.0.0")
    report = validate_data_pack_compatibility(manifest, expectations=_expectations())
    assert report.status.value == "FAIL"


def test_compatibility_rejects_wrong_dataset_checksum() -> None:
    manifest = _manifest(
        source_dataset=SourceDatasetIdentity(
            dataset_name="offers",
            dataset_path="processed/selected_offers.parquet",
            dataset_sha256="tampered",
            dataset_record_count=50,
        )
    )
    report = validate_data_pack_compatibility(manifest, expectations=_expectations())
    assert report.status.value == "FAIL"


def test_shard_index_rejects_gap_ordinals() -> None:
    shard_index = ShardIndex(
        shard_count=2,
        relational_shards=(
            ShardDescriptor(
                ordinal=1,
                relative_path="relational/part-000001.parquet",
                record_count=25,
                sha256="a" * 64,
                source_ref_count=25,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            ),
            ShardDescriptor(
                ordinal=3,
                relative_path="relational/part-000003.parquet",
                record_count=25,
                sha256="c" * 64,
                source_ref_count=25,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            ),
        ),
        embedding_shards=(
            ShardDescriptor(
                ordinal=1,
                relative_path="embeddings/part-000001.parquet",
                record_count=25,
                sha256="b" * 64,
                source_ref_count=25,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            ),
            ShardDescriptor(
                ordinal=3,
                relative_path="embeddings/part-000003.parquet",
                record_count=25,
                sha256="d" * 64,
                source_ref_count=25,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            ),
        ),
    )
    report = validate_shard_index_contract(shard_index)
    assert report.status.value == "FAIL"
