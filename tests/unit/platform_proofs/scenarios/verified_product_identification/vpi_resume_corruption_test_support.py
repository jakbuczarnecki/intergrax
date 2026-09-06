"""Qualification harness support for VPI data pack resume/corruption tests."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.application.catalog.derive_search_representation import (
    build_source_record_ref,
    derive_search_representation,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.wdc_source_offer import (
    parse_wdc_source_offer_json,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
    DataPackBuildReport,
    ShardBuildSeams,
    run_resumable_data_pack_build,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_plan import (
    plan_data_pack_shards,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackShardStatus,
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    semantic_text_hash,
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
    resolve_data_pack_paths,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    FakeDataPackEmbeddingPort,
    patch_canonical_model_identity,
    write_tiny_selected_dataset,
)

QUALIFICATION_ROW_COUNT = 50
QUALIFICATION_SHARD_SIZE = 25
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_DIMENSION = 1024


class ExpectedOutcome(StrEnum):
    RECOVER = "recover"
    FAIL_CLOSED = "fail_closed"


@dataclass(frozen=True, slots=True)
class ReadyShardSnapshot:
    ordinal: int
    relational_sha256: str
    embedding_sha256: str
    relational_digest: str
    embedding_digest: str
    relational_path: Path
    embedding_path: Path
    status: DataPackShardStatus


@dataclass(frozen=True, slots=True)
class DatasetFixture:
    dataset_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class PartialBuildFixture:
    output_root: Path
    dataset: DatasetFixture
    shard_size: int
    row_count: int


class EmbeddingPortFactory(Protocol):
    def __call__(self) -> FakeDataPackEmbeddingPort: ...


def make_dataset(tmp_path: Path, *, row_count: int = QUALIFICATION_ROW_COUNT) -> DatasetFixture:
    dataset_path, manifest_path = write_tiny_selected_dataset(tmp_path / "dataset", row_count=row_count)
    return DatasetFixture(dataset_path=dataset_path, manifest_path=manifest_path)


def build_config(
    fixture: PartialBuildFixture,
    *,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
    max_records: int | None = None,
    shard_size: int | None = None,
) -> DataPackBuildConfig:
    return DataPackBuildConfig(
        output_root=fixture.output_root,
        dataset_path=fixture.dataset.dataset_path,
        dataset_manifest_path=fixture.dataset.manifest_path,
        shard_size=shard_size if shard_size is not None else fixture.shard_size,
        max_records=max_records if max_records is not None else fixture.row_count,
        resume=resume,
        start_fresh=start_fresh,
        stop_after_shard=stop_after_shard,
    )


def run_build(
    fixture: PartialBuildFixture,
    embedding_port: FakeDataPackEmbeddingPort,
    *,
    resume: bool = False,
    start_fresh: bool = False,
    stop_after_shard: int | None = None,
    build_seams: ShardBuildSeams | None = None,
    shard_size: int | None = None,
    max_records: int | None = None,
) -> DataPackBuildReport:
    return run_resumable_data_pack_build(
        build_config(
            fixture,
            resume=resume,
            start_fresh=start_fresh,
            stop_after_shard=stop_after_shard,
            shard_size=shard_size,
            max_records=max_records,
        ),
        embedding_port=embedding_port,
        build_seams=build_seams,
    )


def prepare_partial_build(
    tmp_path: Path,
    monkeypatch,
    *,
    row_count: int = QUALIFICATION_ROW_COUNT,
    shard_size: int = QUALIFICATION_SHARD_SIZE,
    ready_shards: int = 1,
) -> tuple[PartialBuildFixture, FakeDataPackEmbeddingPort]:
    patch_canonical_model_identity(monkeypatch)
    dataset = make_dataset(tmp_path, row_count=row_count)
    fixture = PartialBuildFixture(
        output_root=tmp_path / "pack",
        dataset=dataset,
        shard_size=shard_size,
        row_count=row_count,
    )
    embedding = FakeDataPackEmbeddingPort()
    run_build(fixture, embedding, start_fresh=True, stop_after_shard=ready_shards)
    return fixture, embedding


def read_build_state_dict(fixture: PartialBuildFixture) -> dict[str, object]:
    paths = resolve_data_pack_paths(fixture.output_root)
    return json.loads(paths.build_state_file.read_text(encoding="utf-8"))


def write_build_state_dict(fixture: PartialBuildFixture, payload: dict[str, object]) -> None:
    paths = resolve_data_pack_paths(fixture.output_root)
    paths.build_state_file.write_text(json.dumps(payload), encoding="utf-8")


def shard_dict(fixture: PartialBuildFixture, ordinal: int) -> dict[str, object]:
    payload = read_build_state_dict(fixture)
    shards = payload["shards"]
    if not isinstance(shards, list):
        raise TypeError("shards must be a list")
    for entry in shards:
        if isinstance(entry, dict) and entry.get("ordinal") == ordinal:
            return entry
    raise KeyError(f"shard ordinal {ordinal} not found")


def update_shard(fixture: PartialBuildFixture, ordinal: int, **updates: object) -> None:
    payload = read_build_state_dict(fixture)
    shards = payload["shards"]
    if not isinstance(shards, list):
        raise TypeError("shards must be a list")
    for index, entry in enumerate(shards):
        if isinstance(entry, dict) and entry.get("ordinal") == ordinal:
            shards[index] = {**entry, **updates}
            break
    else:
        raise KeyError(f"shard ordinal {ordinal} not found")
    write_build_state_dict(fixture, payload)


def align_shard_record_count(
    fixture: PartialBuildFixture,
    ordinal: int,
    record_count: int,
) -> None:
    payload = read_build_state_dict(fixture)
    shards_raw = payload["shards"]
    if not isinstance(shards_raw, list):
        raise TypeError("shards must be a list")
    build_total = payload["expected_record_count"]
    shard_size = payload["shard_size"]
    if not isinstance(build_total, int) or not isinstance(shard_size, int):
        raise TypeError("expected_record_count and shard_size must be int")

    sorted_entries = sorted(
        (entry for entry in shards_raw if isinstance(entry, dict)),
        key=lambda entry: int(entry["ordinal"]),
    )
    result: list[dict[str, object]] = []
    start_row_index = 0
    tail_ordinals: list[int] = []

    for entry in sorted_entries:
        entry_ordinal = int(entry["ordinal"])
        if entry_ordinal < ordinal:
            result.append(entry)
            start_row_index = int(entry["end_row_index_exclusive"])
        elif entry_ordinal == ordinal:
            end_row_index_exclusive = start_row_index + record_count
            result.append(
                {
                    **entry,
                    "start_row_index": start_row_index,
                    "end_row_index_exclusive": end_row_index_exclusive,
                    "expected_record_count": record_count,
                }
            )
            start_row_index = end_row_index_exclusive
        else:
            tail_ordinals.append(entry_ordinal)

    if not tail_ordinals:
        payload["shards"] = result
        write_build_state_dict(fixture, payload)
        return

    remaining = build_total - start_row_index
    original_tail = {
        int(entry["ordinal"]): entry
        for entry in sorted_entries
        if int(entry["ordinal"]) in tail_ordinals
    }
    if len(tail_ordinals) == 1:
        original = original_tail[tail_ordinals[0]]
        result.append(
            {
                **original,
                "start_row_index": start_row_index,
                "end_row_index_exclusive": build_total,
                "expected_record_count": remaining,
            }
        )
    else:
        tail_plan = plan_data_pack_shards(record_count=remaining, shard_size=shard_size)
        if len(tail_plan) != len(tail_ordinals):
            msg = (
                f"cannot align tail shard plan: {len(tail_plan)} planned entries "
                f"for {len(tail_ordinals)} tail shards"
            )
            raise ValueError(msg)
        for index, plan_entry in enumerate(tail_plan):
            original = original_tail[tail_ordinals[index]]
            result.append(
                {
                    **original,
                    "start_row_index": start_row_index + plan_entry.start_row_index,
                    "end_row_index_exclusive": start_row_index + plan_entry.end_row_index_exclusive,
                    "expected_record_count": plan_entry.expected_record_count,
                }
            )

    payload["shards"] = result
    write_build_state_dict(fixture, payload)


def snapshot_ready_shard(fixture: PartialBuildFixture, ordinal: int) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    state = read_build_state_file(paths.build_state_file)
    shard = next(entry for entry in state.shards if entry.ordinal == ordinal)
    if shard.relational_relative_path is None or shard.embedding_relative_path is None:
        raise ValueError(f"shard {ordinal} missing relative paths")
    if shard.relational_sha256 is None or shard.embedding_sha256 is None:
        raise ValueError(f"shard {ordinal} missing sha256 metadata")
    if shard.relational_source_ref_set_sha256 is None or shard.embedding_source_ref_set_sha256 is None:
        raise ValueError(f"shard {ordinal} missing source-ref digests")
    return ReadyShardSnapshot(
        ordinal=ordinal,
        relational_sha256=shard.relational_sha256,
        embedding_sha256=shard.embedding_sha256,
        relational_digest=shard.relational_source_ref_set_sha256,
        embedding_digest=shard.embedding_source_ref_set_sha256,
        relational_path=paths.root / shard.relational_relative_path,
        embedding_path=paths.root / shard.embedding_relative_path,
        status=shard.status,
    )


def flip_file_byte(path: Path, offset: int = 10) -> bytes:
    data = bytearray(path.read_bytes())
    original = bytes([data[offset]])
    data[offset] ^= 0xFF
    path.write_bytes(data)
    return original


def delete_file(path: Path) -> None:
    path.unlink()


def _offer_json(offer_id: str) -> str:
    index = int(offer_id.split("-")[-1])
    return json.dumps(
        {
            "id": offer_id,
            "title": f"Relay module {index}",
            "identifiers": [{"gtin": f"{1000000000000 + index}"}],
            "keyValuePairs": {"voltage": "24V"},
        }
    )


def _relational_record(
    *,
    global_row_index: int,
    offer_id: str,
    catalog_id: str = "wdc-v2-selected",
) -> RelationalDataPackRecord:
    sample_json = _offer_json(offer_id)
    source_offer = parse_wdc_source_offer_json(sample_json)
    source_ref = build_source_record_ref(source_offer, catalog_id=catalog_id)
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    semantic_text = representation.semantic.semantic_text
    return RelationalDataPackRecord(
        global_row_index=global_row_index,
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


def _embedding_record(relational: RelationalDataPackRecord) -> EmbeddingDataPackRecord:
    vector = tuple(0.01 * index for index in range(EMBEDDING_DIMENSION))
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
        embedding_model_revision=MODEL_REVISION,
        embedding_dimension=EMBEDDING_DIMENSION,
        dense_embedding=vector,
    )


def write_valid_shard_pair(
    paths_root: Path,
    ordinal: int,
    records: tuple[RelationalDataPackRecord, ...],
) -> tuple[str, str, str]:
    relational_path = final_shard_path(paths_root / "relational", ordinal)
    embedding_path = final_shard_path(paths_root / "embeddings", ordinal)
    relational_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_records = tuple(_embedding_record(record) for record in records)
    write_relational_parquet(relational_path, records)
    write_embedding_parquet(
        embedding_path,
        embedding_records,
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    digest = source_ref_set_sha256(tuple(record.source_ref for record in records))
    return sha256_file(relational_path), sha256_file(embedding_path), digest


def write_schema_corrupt_relational(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({"wrong_column": pa.array([1], type=pa.int64())})
    pq.write_table(table, path)
    return sha256_file(path)


def write_dimension_corrupt_embedding(path: Path, relational: RelationalDataPackRecord) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    vector = [0.1] * EMBEDDING_DIMENSION
    table = pa.table(
        {
            "logical_point_id": pa.array([search_representation_point_id(
                catalog_id=relational.source_ref.catalog_id,
                offer_id=relational.source_ref.offer_id.value,
                derivation_version=relational.derivation_version,
            )], type=pa.string()),
            "catalog_id": pa.array([relational.source_ref.catalog_id], type=pa.string()),
            "offer_id": pa.array([relational.source_ref.offer_id.value], type=pa.string()),
            "source_revision": pa.array([relational.source_ref.source_revision], type=pa.string()),
            "derivation_version": pa.array([relational.derivation_version], type=pa.string()),
            "semantic_text_hash": pa.array([relational.semantic_text_hash], type=pa.string()),
            "embedding_provider": pa.array(["hf"], type=pa.string()),
            "embedding_model": pa.array(["BAAI/bge-m3"], type=pa.string()),
            "embedding_model_revision": pa.array([MODEL_REVISION], type=pa.string()),
            "embedding_dimension": pa.array([512], type=pa.int32()),
            "dense_embedding": pa.array([vector], type=pa.list_(pa.float32(), EMBEDDING_DIMENSION)),
        }
    )
    pq.write_table(table, path)
    return sha256_file(path)


def install_ready_shard_from_builder(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    return snapshot_ready_shard(fixture, ordinal)


def corrupt_ready_shard_metadata_digest(
    fixture: PartialBuildFixture,
    ordinal: int,
    *,
    relational_digest: str | None = None,
    embedding_digest: str | None = None,
) -> ReadyShardSnapshot:
    snapshot = snapshot_ready_shard(fixture, ordinal)
    updates: dict[str, object] = {}
    if relational_digest is not None:
        updates["relational_source_ref_set_sha256"] = relational_digest
    if embedding_digest is not None:
        updates["embedding_source_ref_set_sha256"] = embedding_digest
    update_shard(fixture, ordinal, **updates)
    return snapshot


def install_pair_mismatch_ready_shard(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    record_a = _relational_record(global_row_index=0, offer_id="offer-9001")
    record_b = _relational_record(global_row_index=0, offer_id="offer-9002")
    relational_path = final_shard_path(paths.relational_dir, ordinal)
    embedding_path = final_shard_path(paths.embeddings_dir, ordinal)
    relational_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    write_relational_parquet(relational_path, (record_a,))
    write_embedding_parquet(
        embedding_path,
        (_embedding_record(record_b),),
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    relational_sha = sha256_file(relational_path)
    embedding_sha = sha256_file(embedding_path)
    digest_a = source_ref_set_sha256((record_a.source_ref,))
    digest_b = source_ref_set_sha256((record_b.source_ref,))
    update_shard(
        fixture,
        ordinal,
        status=DataPackShardStatus.READY.value,
        relational_relative_path=f"relational/part-{ordinal:06d}.parquet",
        embedding_relative_path=f"embeddings/part-{ordinal:06d}.parquet",
        relational_sha256=relational_sha,
        embedding_sha256=embedding_sha,
        relational_source_ref_set_sha256=digest_a,
        embedding_source_ref_set_sha256=digest_b,
        attempt=1,
    )
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = sum(
        1
        for entry in payload["shards"]
        if isinstance(entry, dict) and entry.get("status") == DataPackShardStatus.READY.value
    )
    write_build_state_dict(fixture, payload)
    return snapshot_ready_shard(fixture, ordinal)


def install_duplicate_source_ref_ready_shard(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    record = _relational_record(global_row_index=0, offer_id="offer-9005")
    duplicate = _relational_record(global_row_index=1, offer_id="offer-9005")
    relational_sha, embedding_sha, digest = write_valid_shard_pair(
        paths.root,
        ordinal,
        (record, duplicate),
    )
    update_shard(
        fixture,
        ordinal,
        status=DataPackShardStatus.READY.value,
        relational_relative_path=f"relational/part-{ordinal:06d}.parquet",
        embedding_relative_path=f"embeddings/part-{ordinal:06d}.parquet",
        relational_sha256=relational_sha,
        embedding_sha256=embedding_sha,
        relational_source_ref_set_sha256=digest,
        embedding_source_ref_set_sha256=digest,
        expected_record_count=2,
        attempt=1,
    )
    align_shard_record_count(fixture, ordinal, 2)
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = sum(
        1
        for entry in payload["shards"]
        if isinstance(entry, dict) and entry.get("status") == DataPackShardStatus.READY.value
    )
    write_build_state_dict(fixture, payload)
    return snapshot_ready_shard(fixture, ordinal)


def install_record_count_mismatch_ready_shard(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    record = _relational_record(global_row_index=0, offer_id="offer-9006")
    relational_sha, embedding_sha, digest = write_valid_shard_pair(paths.root, ordinal, (record,))
    update_shard(
        fixture,
        ordinal,
        status=DataPackShardStatus.READY.value,
        relational_relative_path=f"relational/part-{ordinal:06d}.parquet",
        embedding_relative_path=f"embeddings/part-{ordinal:06d}.parquet",
        relational_sha256=relational_sha,
        embedding_sha256=embedding_sha,
        relational_source_ref_set_sha256=digest,
        embedding_source_ref_set_sha256=digest,
        expected_record_count=2,
        attempt=1,
    )
    align_shard_record_count(fixture, ordinal, 2)
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = sum(
        1
        for entry in payload["shards"]
        if isinstance(entry, dict) and entry.get("status") == DataPackShardStatus.READY.value
    )
    write_build_state_dict(fixture, payload)
    return snapshot_ready_shard(fixture, ordinal)


def install_schema_corrupt_ready_shard(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    relational_path = final_shard_path(paths.relational_dir, ordinal)
    embedding_path = final_shard_path(paths.embeddings_dir, ordinal)
    record = _relational_record(global_row_index=0, offer_id="offer-9003")
    relational_sha = write_schema_corrupt_relational(relational_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    write_embedding_parquet(
        embedding_path,
        (_embedding_record(record),),
        embedding_dimension=EMBEDDING_DIMENSION,
    )
    embedding_sha = sha256_file(embedding_path)
    digest = source_ref_set_sha256((record.source_ref,))
    update_shard(
        fixture,
        ordinal,
        status=DataPackShardStatus.READY.value,
        relational_relative_path=f"relational/part-{ordinal:06d}.parquet",
        embedding_relative_path=f"embeddings/part-{ordinal:06d}.parquet",
        relational_sha256=relational_sha,
        embedding_sha256=embedding_sha,
        relational_source_ref_set_sha256=digest,
        embedding_source_ref_set_sha256=digest,
        expected_record_count=1,
        attempt=1,
    )
    align_shard_record_count(fixture, ordinal, 1)
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = 1
    write_build_state_dict(fixture, payload)
    return snapshot_ready_shard(fixture, ordinal)


def install_dimension_corrupt_ready_shard(
    fixture: PartialBuildFixture,
    ordinal: int,
) -> ReadyShardSnapshot:
    paths = resolve_data_pack_paths(fixture.output_root)
    record = _relational_record(global_row_index=0, offer_id="offer-9004")
    relational_sha, _, digest = write_valid_shard_pair(paths.root, ordinal, (record,))
    embedding_path = final_shard_path(paths.embeddings_dir, ordinal)
    embedding_sha = write_dimension_corrupt_embedding(embedding_path, record)
    update_shard(
        fixture,
        ordinal,
        status=DataPackShardStatus.READY.value,
        relational_relative_path=f"relational/part-{ordinal:06d}.parquet",
        embedding_relative_path=f"embeddings/part-{ordinal:06d}.parquet",
        relational_sha256=relational_sha,
        embedding_sha256=embedding_sha,
        relational_source_ref_set_sha256=digest,
        embedding_source_ref_set_sha256=digest,
        expected_record_count=1,
        attempt=1,
    )
    align_shard_record_count(fixture, ordinal, 1)
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = 1
    write_build_state_dict(fixture, payload)
    return snapshot_ready_shard(fixture, ordinal)


def setup_non_ready_state(
    fixture: PartialBuildFixture,
    *,
    target_ordinal: int,
    status: DataPackShardStatus,
    ready_shards: int = 1,
) -> None:
    paths = resolve_data_pack_paths(fixture.output_root)
    update_shard(
        fixture,
        target_ordinal,
        status=status.value,
        relational_relative_path=None,
        embedding_relative_path=None,
        relational_sha256=None,
        embedding_sha256=None,
        relational_source_ref_set_sha256=None,
        embedding_source_ref_set_sha256=None,
        attempt=1,
    )
    payload = read_build_state_dict(fixture)
    payload["completed_shards"] = ready_shards
    write_build_state_dict(fixture, payload)
    discard = (
        temp_shard_path(paths.relational_dir, target_ordinal),
        temp_shard_path(paths.embeddings_dir, target_ordinal),
        final_shard_path(paths.relational_dir, target_ordinal),
        final_shard_path(paths.embeddings_dir, target_ordinal),
    )
    for path in discard:
        if path.exists():
            path.unlink()


def assert_ready_shard_immutable(
    before: ReadyShardSnapshot,
    after: ReadyShardSnapshot,
) -> None:
    assert after.status is DataPackShardStatus.READY
    assert after.relational_sha256 == before.relational_sha256
    assert after.embedding_sha256 == before.embedding_sha256
    assert after.relational_digest == before.relational_digest
    assert after.embedding_digest == before.embedding_digest
    assert sha256_file(after.relational_path) == before.relational_sha256
    assert sha256_file(after.embedding_path) == before.embedding_sha256


def assert_no_temp_files(fixture: PartialBuildFixture) -> None:
    paths = resolve_data_pack_paths(fixture.output_root)
    temps = list(paths.relational_dir.glob("*.tmp")) + list(paths.embeddings_dir.glob("*.tmp"))
    assert temps == []


def assert_not_distributable(fixture: PartialBuildFixture) -> None:
    paths = resolve_data_pack_paths(fixture.output_root)
    assert not paths.manifest_file.exists() or _manifest_not_ready(paths.manifest_file)
    assert not paths.shards_index_file.exists()
    assert not paths.checksums_file.exists()


def _manifest_not_ready(manifest_path: Path) -> bool:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return payload.get("status") != "READY"


def apply_non_ready_mutation(
    fixture: PartialBuildFixture,
    *,
    target_ordinal: int,
    status: DataPackShardStatus,
    mutation: str,
) -> None:
    paths = resolve_data_pack_paths(fixture.output_root)
    setup_non_ready_state(fixture, target_ordinal=target_ordinal, status=status)
    rel_temp = temp_shard_path(paths.relational_dir, target_ordinal)
    emb_temp = temp_shard_path(paths.embeddings_dir, target_ordinal)
    rel_final = final_shard_path(paths.relational_dir, target_ordinal)
    emb_final = final_shard_path(paths.embeddings_dir, target_ordinal)
    rel_rel = f"relational/part-{target_ordinal:06d}.parquet"
    emb_rel = f"embeddings/part-{target_ordinal:06d}.parquet"

    if mutation == "none":
        return
    if mutation == "orphan_tmp":
        rel_temp.write_text("tmp", encoding="utf-8")
        emb_temp.write_text("tmp", encoding="utf-8")
        return
    if mutation == "orphan_final":
        rel_final.write_text("orphan", encoding="utf-8")
        return
    if mutation == "writing_rel_tmp":
        rel_temp.write_text("partial", encoding="utf-8")
        return
    if mutation == "writing_both_tmp":
        rel_temp.write_text("partial-rel", encoding="utf-8")
        emb_temp.write_text("partial-emb", encoding="utf-8")
        return
    if mutation == "writing_malformed_tmp":
        rel_temp.write_bytes(b"not-parquet")
        emb_temp.write_bytes(b"not-parquet")
        return
    if mutation == "validating_both_tmp":
        rel_temp.write_bytes(b"tmp-rel")
        emb_temp.write_bytes(b"tmp-emb")
        return
    if mutation == "validating_rel_final_emb_tmp":
        rel_final.write_bytes(b"orphan-rel")
        emb_temp.write_bytes(b"tmp-emb")
        update_shard(
            fixture,
            target_ordinal,
            relational_relative_path=rel_rel,
            embedding_relative_path=emb_rel,
        )
        return
    if mutation == "validating_rel_final_only":
        rel_final.write_bytes(b"orphan-rel")
        update_shard(
            fixture,
            target_ordinal,
            relational_relative_path=rel_rel,
            embedding_relative_path=emb_rel,
        )
        return
    if mutation == "validating_emb_final_only":
        emb_final.write_bytes(b"orphan-emb")
        update_shard(
            fixture,
            target_ordinal,
            relational_relative_path=rel_rel,
            embedding_relative_path=emb_rel,
        )
        return
    if mutation == "validating_both_finals":
        rel_final.write_bytes(b"orphan-rel")
        emb_final.write_bytes(b"orphan-emb")
        update_shard(
            fixture,
            target_ordinal,
            relational_relative_path=rel_rel,
            embedding_relative_path=emb_rel,
        )
        return
    raise ValueError(f"unknown mutation: {mutation}")


def resume_and_finalize(
    fixture: PartialBuildFixture,
    embedding_port: FakeDataPackEmbeddingPort,
) -> DataPackBuildReport:
    return run_build(fixture, embedding_port, resume=True)
