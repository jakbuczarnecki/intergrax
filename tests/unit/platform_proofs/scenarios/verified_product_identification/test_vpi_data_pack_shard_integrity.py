"""Shard integrity, source-ref digest, and strict parser tests."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.identifiers import (
    ProductOfferId,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    verify_sha256sums,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility import (
    validate_data_pack_compatibility,
    validate_shard_index_contract,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_integrity import (
    validate_shard_pair_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackFormatError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    source_ref_set_sha256,
    source_ref_set_sha256_from_keys,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
    ShardIndex,
    shard_index_from_json_dict,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_vpi_data_pack_compatibility import (
    _expectations,
    _manifest,
)

pytestmark = pytest.mark.unit

_FIXTURES = Path(__file__).resolve().parents[4] / "fixtures"
_DIGEST_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_DIGEST_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
_DIGEST_C = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"


def _source_ref(*, offer_id: str, revision: str | None = None) -> SourceRecordRef:
    return SourceRecordRef(
        offer_id=ProductOfferId(offer_id),
        catalog_id="wdc-v2-selected",
        source_revision=revision,
    )


def _descriptor(
    *,
    ordinal: int,
    relative_path: str,
    record_count: int,
    sha256: str,
    source_ref_set_sha256: str,
    schema_version: str,
) -> ShardDescriptor:
    return ShardDescriptor(
        ordinal=ordinal,
        relative_path=relative_path,
        record_count=record_count,
        sha256=sha256,
        source_ref_count=record_count,
        source_ref_set_sha256=source_ref_set_sha256,
        schema_version=schema_version,
    )


def test_source_ref_set_sha256_same_set_different_order() -> None:
    refs_a = (_source_ref(offer_id="a"), _source_ref(offer_id="b"))
    refs_b = (_source_ref(offer_id="b"), _source_ref(offer_id="a"))
    assert source_ref_set_sha256(refs_a) == source_ref_set_sha256(refs_b)


def test_source_ref_set_sha256_different_source_ref() -> None:
    left = source_ref_set_sha256((_source_ref(offer_id="a"),))
    right = source_ref_set_sha256((_source_ref(offer_id="b"),))
    assert left != right


def test_source_ref_set_sha256_source_revision_difference() -> None:
    left = source_ref_set_sha256((_source_ref(offer_id="a", revision="r1"),))
    right = source_ref_set_sha256((_source_ref(offer_id="a", revision="r2"),))
    assert left != right


def test_source_ref_set_sha256_from_keys_is_deterministic() -> None:
    keys = (("cat", "offer-1", None), ("cat", "offer-2", "rev"))
    assert source_ref_set_sha256_from_keys(keys) == source_ref_set_sha256_from_keys(reversed(keys))


def test_shard_pairing_same_count_different_refs_fails() -> None:
    relational = _descriptor(
        ordinal=1,
        relative_path="relational/part-000001.parquet",
        record_count=2,
        sha256=_DIGEST_A,
        source_ref_set_sha256=_DIGEST_A,
        schema_version=RELATIONAL_SCHEMA_VERSION,
    )
    embedding = _descriptor(
        ordinal=1,
        relative_path="embeddings/part-000001.parquet",
        record_count=2,
        sha256=_DIGEST_B,
        source_ref_set_sha256=_DIGEST_B,
        schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    report = validate_shard_index_contract(
        ShardIndex(shard_count=1, relational_shards=(relational,), embedding_shards=(embedding,))
    )
    assert report.status.value == "FAIL"


def test_shard_pairing_same_refs_different_order_passes() -> None:
    digest = source_ref_set_sha256((_source_ref(offer_id="a"), _source_ref(offer_id="b")))
    relational = _descriptor(
        ordinal=1,
        relative_path="relational/part-000001.parquet",
        record_count=2,
        sha256=_DIGEST_A,
        source_ref_set_sha256=digest,
        schema_version=RELATIONAL_SCHEMA_VERSION,
    )
    embedding = _descriptor(
        ordinal=1,
        relative_path="embeddings/part-000001.parquet",
        record_count=2,
        sha256=_DIGEST_B,
        source_ref_set_sha256=digest,
        schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    checks = validate_shard_pair_identity(relational, embedding)
    assert all(check.status.value == "PASS" for check in checks)


def test_shard_pairing_different_count_fails() -> None:
    relational = _descriptor(
        ordinal=1,
        relative_path="relational/part-000001.parquet",
        record_count=2,
        sha256=_DIGEST_A,
        source_ref_set_sha256=_DIGEST_C,
        schema_version=RELATIONAL_SCHEMA_VERSION,
    )
    embedding = _descriptor(
        ordinal=1,
        relative_path="embeddings/part-000001.parquet",
        record_count=3,
        sha256=_DIGEST_B,
        source_ref_set_sha256=_DIGEST_C,
        schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    report = validate_shard_index_contract(
        ShardIndex(shard_count=1, relational_shards=(relational,), embedding_shards=(embedding,))
    )
    assert report.status.value == "FAIL"


def test_shard_index_rejects_duplicate_ordinal() -> None:
    payload = {
        "shard_count": 2,
        "relational_shards": [
            {
                "ordinal": 1,
                "relative_path": "relational/part-000001.parquet",
                "record_count": 1,
                "sha256": _DIGEST_A,
                "source_ref_count": 1,
                "source_ref_set_sha256": _DIGEST_C,
                "schema_version": RELATIONAL_SCHEMA_VERSION,
            },
            {
                "ordinal": 1,
                "relative_path": "relational/part-000002.parquet",
                "record_count": 1,
                "sha256": _DIGEST_B,
                "source_ref_count": 1,
                "source_ref_set_sha256": _DIGEST_C,
                "schema_version": RELATIONAL_SCHEMA_VERSION,
            },
        ],
        "embedding_shards": [
            {
                "ordinal": 1,
                "relative_path": "embeddings/part-000001.parquet",
                "record_count": 2,
                "sha256": _DIGEST_A,
                "source_ref_count": 2,
                "source_ref_set_sha256": _DIGEST_C,
                "schema_version": EMBEDDING_SCHEMA_VERSION,
            },
            {
                "ordinal": 2,
                "relative_path": "embeddings/part-000002.parquet",
                "record_count": 1,
                "sha256": _DIGEST_B,
                "source_ref_count": 1,
                "source_ref_set_sha256": _DIGEST_C,
                "schema_version": EMBEDDING_SCHEMA_VERSION,
            },
        ],
    }
    with pytest.raises(VpiDataPackFormatError):
        shard_index_from_json_dict(payload)


def test_strict_parser_rejects_missing_source_ref_set_sha256() -> None:
    payload = {
        "ordinal": 1,
        "relative_path": "relational/part-000001.parquet",
        "record_count": 1,
        "sha256": _DIGEST_A,
        "source_ref_count": 1,
        "schema_version": RELATIONAL_SCHEMA_VERSION,
    }
    with pytest.raises(VpiDataPackFormatError):
        shard_index_from_json_dict(
            {
                "shard_count": 1,
                "relational_shards": [payload],
                "embedding_shards": [payload | {"source_ref_set_sha256": _DIGEST_C}],
            }
        )


def test_strict_parser_rejects_malformed_source_ref_set_sha256() -> None:
    payload = {
        "ordinal": 1,
        "relative_path": "relational/part-000001.parquet",
        "record_count": 1,
        "sha256": _DIGEST_A,
        "source_ref_count": 1,
        "source_ref_set_sha256": "abc",
        "schema_version": RELATIONAL_SCHEMA_VERSION,
    }
    with pytest.raises(VpiDataPackFormatError):
        shard_index_from_json_dict(
            {
                "shard_count": 1,
                "relational_shards": [payload],
                "embedding_shards": [payload],
            }
        )


def test_strict_parser_rejects_wrong_type_source_ref_set_sha256() -> None:
    payload = {
        "ordinal": 1,
        "relative_path": "relational/part-000001.parquet",
        "record_count": 1,
        "sha256": _DIGEST_A,
        "source_ref_count": 1,
        "source_ref_set_sha256": 123,
        "schema_version": RELATIONAL_SCHEMA_VERSION,
    }
    with pytest.raises(VpiDataPackFormatError):
        shard_index_from_json_dict(
            {
                "shard_count": 1,
                "relational_shards": [payload],
                "embedding_shards": [payload],
            }
        )


def test_golden_shard_index_fixture_parses() -> None:
    payload = json.loads((_FIXTURES / "vpi_data_pack_shard_index_v1.json").read_text(encoding="utf-8"))
    shard_index = shard_index_from_json_dict(payload)
    assert shard_index.shard_count == 1
    assert (
        shard_index.relational_shards[0].source_ref_set_sha256
        == shard_index.embedding_shards[0].source_ref_set_sha256
    )


def test_checksum_mismatch_is_typed_integrity_failure(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"proof-50")
    checksums = tmp_path / "SHA256SUMS"
    write_sha256sums(checksums, (("artifact.bin", target),))
    target.write_bytes(b"tampered")
    with pytest.raises(VpiDataPackIntegrityError):
        verify_sha256sums(checksums, tmp_path)


def test_checksum_validation_does_not_swallow_unexpected_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest()
    pack_root = tmp_path / "pack"
    indexes = pack_root / "indexes"
    checksums = pack_root / "checksums"
    indexes.mkdir(parents=True)
    checksums.mkdir(parents=True)
    (indexes / "shards.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "relational_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "relational/part-000001.parquet",
                        "record_count": 50,
                        "sha256": _DIGEST_A,
                        "source_ref_count": 50,
                        "source_ref_set_sha256": _DIGEST_C,
                        "schema_version": RELATIONAL_SCHEMA_VERSION,
                    }
                ],
                "embedding_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "embeddings/part-000001.parquet",
                        "record_count": 50,
                        "sha256": _DIGEST_B,
                        "source_ref_count": 50,
                        "source_ref_set_sha256": _DIGEST_C,
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (checksums / "SHA256SUMS").write_text("deadbeef\n", encoding="utf-8")

    def _boom(_path: Path, _root: Path) -> None:
        raise RuntimeError("unexpected programmer error")

    monkeypatch.setattr(
        "platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.compatibility.verify_sha256sums",
        _boom,
    )
    with pytest.raises(RuntimeError, match="unexpected programmer error"):
        validate_data_pack_compatibility(manifest, expectations=_expectations(), pack_root=pack_root)


def _build_single_record_pack(tmp_path: Path) -> tuple[Path, str, str, str]:
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
        EmbeddingDataPackRecord,
    )
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
        RelationalDataPackRecord,
    )
    from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
        semantic_text_hash,
    )
    from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
        search_representation_point_id,
    )

    sample_json = json.dumps({"id": "offer-1", "title": "Widget"})
    source_offer = parse_wdc_source_offer_json(sample_json)
    source_ref = build_source_record_ref(source_offer, catalog_id="wdc-v2-selected")
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    semantic_text = representation.semantic.semantic_text
    relational = RelationalDataPackRecord(
        global_row_index=0,
        source_ref=source_ref,
        record_json=sample_json,
        derivation_version=representation.derivation_version,
        semantic_text=semantic_text,
        semantic_text_hash=semantic_text_hash(semantic_text),
        title=source_offer.title,
        brand=source_offer.brand,
        category=source_offer.category,
        description=source_offer.description,
        has_identifiers=False,
        has_spec_table=False,
        has_structured_attributes=False,
    )
    vector = tuple(0.01 * index for index in range(1024))
    embedding = EmbeddingDataPackRecord(
        logical_point_id=search_representation_point_id(
            catalog_id=source_ref.catalog_id,
            offer_id=source_ref.offer_id.value,
            derivation_version=relational.derivation_version,
        ),
        source_ref=source_ref,
        derivation_version=relational.derivation_version,
        semantic_text_hash=relational.semantic_text_hash,
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_model_revision="5617a9f61b028005a4858fdac845db406aefb181",
        embedding_dimension=1024,
        dense_embedding=vector,
    )
    relational_dir = tmp_path / "relational"
    embedding_dir = tmp_path / "embeddings"
    relational_dir.mkdir()
    embedding_dir.mkdir()
    relational_path = relational_dir / "part-000001.parquet"
    embedding_path = embedding_dir / "part-000001.parquet"
    write_relational_parquet(relational_path, (relational,))
    write_embedding_parquet(embedding_path, (embedding,), embedding_dimension=1024)
    digest = source_ref_set_sha256((source_ref,))
    return tmp_path, sha256_file(relational_path), sha256_file(embedding_path), digest


def test_compatibility_validates_relational_and_embedding_file_sha(tmp_path: Path) -> None:
    pack_root, relational_sha, embedding_sha, digest = _build_single_record_pack(tmp_path)
    manifest = _manifest(record_count=1, shard_count=1, source_record_count=1)
    indexes = pack_root / "indexes"
    checksums = pack_root / "checksums"
    indexes.mkdir(exist_ok=True)
    checksums.mkdir(exist_ok=True)
    manifest_path = pack_root / "manifest/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (indexes / "shards.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "relational_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "relational/part-000001.parquet",
                        "record_count": 1,
                        "sha256": relational_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": RELATIONAL_SCHEMA_VERSION,
                    }
                ],
                "embedding_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "embeddings/part-000001.parquet",
                        "record_count": 1,
                        "sha256": embedding_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_sha256sums(
        checksums / "SHA256SUMS",
        (
            ("manifest/manifest.json", manifest_path),
            ("relational/part-000001.parquet", pack_root / "relational/part-000001.parquet"),
            ("embeddings/part-000001.parquet", pack_root / "embeddings/part-000001.parquet"),
            ("indexes/shards.json", indexes / "shards.json"),
        ),
    )
    report = validate_data_pack_compatibility(
        manifest,
        expectations=_expectations(source_dataset_sha256="abc"),
        pack_root=pack_root,
    )
    sha_checks = [
        check
        for check in report.checks
        if check.name.startswith(("relational_shard_sha256_", "embedding_shard_sha256_"))
    ]
    assert len(sha_checks) == 2
    assert all(check.status.value == "PASS" for check in sha_checks)


def test_compatibility_rejects_tampered_relational_file(tmp_path: Path) -> None:
    pack_root, relational_sha, embedding_sha, digest = _build_single_record_pack(tmp_path)
    relational_path = pack_root / "relational/part-000001.parquet"
    relational_path.write_bytes(relational_path.read_bytes() + b"tamper")
    manifest = _manifest(record_count=1, shard_count=1, source_record_count=1)
    indexes = pack_root / "indexes"
    checksums = pack_root / "checksums"
    indexes.mkdir(exist_ok=True)
    checksums.mkdir(exist_ok=True)
    manifest_path = pack_root / "manifest/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (indexes / "shards.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "relational_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "relational/part-000001.parquet",
                        "record_count": 1,
                        "sha256": relational_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": RELATIONAL_SCHEMA_VERSION,
                    }
                ],
                "embedding_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "embeddings/part-000001.parquet",
                        "record_count": 1,
                        "sha256": embedding_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_sha256sums(
        checksums / "SHA256SUMS",
        (("indexes/shards.json", indexes / "shards.json"),),
    )
    report = validate_data_pack_compatibility(
        manifest,
        expectations=_expectations(source_dataset_sha256="abc"),
        pack_root=pack_root,
    )
    assert any(
        check.name == "relational_shard_sha256_1" and check.status.value == "FAIL"
        for check in report.checks
    )


def test_compatibility_rejects_missing_embedding_file(tmp_path: Path) -> None:
    pack_root, relational_sha, embedding_sha, digest = _build_single_record_pack(tmp_path)
    embedding_path = pack_root / "embeddings/part-000001.parquet"
    embedding_path.unlink()
    manifest = _manifest(record_count=1, shard_count=1, source_record_count=1)
    indexes = pack_root / "indexes"
    checksums = pack_root / "checksums"
    indexes.mkdir(exist_ok=True)
    checksums.mkdir(exist_ok=True)
    manifest_path = pack_root / "manifest/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    (indexes / "shards.json").write_text(
        json.dumps(
            {
                "shard_count": 1,
                "relational_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "relational/part-000001.parquet",
                        "record_count": 1,
                        "sha256": relational_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": RELATIONAL_SCHEMA_VERSION,
                    }
                ],
                "embedding_shards": [
                    {
                        "ordinal": 1,
                        "relative_path": "embeddings/part-000001.parquet",
                        "record_count": 1,
                        "sha256": embedding_sha,
                        "source_ref_count": 1,
                        "source_ref_set_sha256": digest,
                        "schema_version": EMBEDDING_SCHEMA_VERSION,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    write_sha256sums(
        checksums / "SHA256SUMS",
        (("indexes/shards.json", indexes / "shards.json"),),
    )
    report = validate_data_pack_compatibility(
        manifest,
        expectations=_expectations(source_dataset_sha256="abc"),
        pack_root=pack_root,
    )
    assert any(
        check.name == "embedding_shard_missing_1" and check.status.value == "FAIL"
        for check in report.checks
    )


def test_integrity_modules_have_no_provider_imports() -> None:
    module_paths = (
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/contracts/identity.py",
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/contracts/shard_index.py",
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/contracts/json_decode.py",
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/shard_integrity.py",
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/checksums.py",
        Path(__file__).resolve().parents[5]
        / "platform_proofs/scenarios/verified_product_identification/dataset/data_pack/application/compatibility.py",
    )
    violations: list[str] = []
    for path in module_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and ".integrations." in node.module:
                violations.append(f"{path.name}:{node.module}")
    assert violations == []
