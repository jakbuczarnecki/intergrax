"""Reader, streaming batching, and bounded-memory tests for storage bootstrap."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
    write_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    semantic_text_hash,
    source_ref_set_sha256,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    write_shard_index_file,
    ShardDescriptor,
    ShardIndex,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    write_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    write_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    iter_record_batches,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchSize,
    BootstrapFinalStatus,
    BootstrapRequest,
    RelationalTargetId,
    ResumeMode,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    StorageBootstrapIdentityError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.errors import (
    DataPackReaderIntegrityError,
    DataPackReaderOrderingError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.reader.filesystem_reader import (
    FilesystemDataPackBootstrapReader,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.test_storage_bootstrap_data_pack_load import (
    FakeRelationalAdapter,
    FakeVectorAdapter,
    InMemoryBootstrapCheckpointStore,
    _build_embedding,
    _build_pairs,
    _build_relational,
    _ready_manifest,
)

pytestmark = pytest.mark.unit

_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_EMBEDDING_DIMENSION = 8
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATA_PACK_LOAD_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load"
)


def _write_ready_pack(
    tmp_path: Path,
    *,
    record_count: int,
    shard_size: int = 3,
    status: DataPackStatus = DataPackStatus.READY,
) -> Path:
    pairs = _build_pairs(record_count)
    relational_shards: list[ShardDescriptor] = []
    embedding_shards: list[ShardDescriptor] = []
    shard_count = (record_count + shard_size - 1) // shard_size if record_count else 0
    if shard_count == 0:
        shard_count = 1
    for ordinal in range(1, shard_count + 1):
        start = (ordinal - 1) * shard_size
        end = min(start + shard_size, record_count)
        shard_pairs = pairs[start:end]
        if not shard_pairs:
            continue
        relational_path = final_shard_path(tmp_path / "relational", ordinal)
        embedding_path = final_shard_path(tmp_path / "embeddings", ordinal)
        relational_path.parent.mkdir(parents=True, exist_ok=True)
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        relational_records = tuple(pair.relational for pair in shard_pairs)
        embedding_records = tuple(pair.embedding for pair in shard_pairs)
        write_relational_parquet(relational_path, relational_records)
        write_embedding_parquet(
            embedding_path,
            embedding_records,
            embedding_dimension=_EMBEDDING_DIMENSION,
        )
        digest = source_ref_set_sha256(tuple(record.source_ref for record in relational_records))
        relational_shards.append(
            ShardDescriptor(
                ordinal=ordinal,
                relative_path=f"relational/part-{ordinal:06d}.parquet",
                record_count=len(relational_records),
                sha256=sha256_file(relational_path),
                source_ref_count=len(relational_records),
                source_ref_set_sha256=digest,
                schema_version=RELATIONAL_SCHEMA_VERSION,
            )
        )
        embedding_shards.append(
            ShardDescriptor(
                ordinal=ordinal,
                relative_path=f"embeddings/part-{ordinal:06d}.parquet",
                record_count=len(embedding_records),
                sha256=sha256_file(embedding_path),
                source_ref_count=len(embedding_records),
                source_ref_set_sha256=digest,
                schema_version=EMBEDDING_SCHEMA_VERSION,
            )
        )
    manifest = _ready_manifest(record_count)
    manifest = DataPackManifest(
        data_pack_version=manifest.data_pack_version,
        content_identity=manifest.content_identity,
        scenario_id=manifest.scenario_id,
        source_dataset=manifest.source_dataset,
        source_record_count=manifest.source_record_count,
        sample_identity=manifest.sample_identity,
        derivation_version=manifest.derivation_version,
        semantic_text_version=manifest.semantic_text_version,
        embedding_identity=manifest.embedding_identity,
        relational_schema_version=manifest.relational_schema_version,
        embedding_schema_version=manifest.embedding_schema_version,
        relational_format=manifest.relational_format,
        embedding_format=manifest.embedding_format,
        shard_count=len(relational_shards),
        record_count=record_count,
        created_at_utc=manifest.created_at_utc,
        status=status,
        checksums_path=manifest.checksums_path,
        shards_index_path=manifest.shards_index_path,
        build_execution_provenance=manifest.build_execution_provenance,
    )
    manifest_path = tmp_path / "manifest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_manifest_file(manifest_path, manifest)
    shard_index_path = tmp_path / "indexes" / "shards.json"
    write_shard_index_file(
        shard_index_path,
        ShardIndex(
            shard_count=len(relational_shards),
            relational_shards=tuple(relational_shards),
            embedding_shards=tuple(embedding_shards),
        ),
    )
    checksum_entries: list[tuple[str, Path]] = [
        ("manifest/manifest.json", manifest_path),
        ("indexes/shards.json", shard_index_path),
    ]
    for descriptor in relational_shards:
        checksum_entries.append(
            (descriptor.relative_path, tmp_path / descriptor.relative_path)
        )
    for descriptor in embedding_shards:
        checksum_entries.append(
            (descriptor.relative_path, tmp_path / descriptor.relative_path)
        )
    checksums_dir = tmp_path / "checksums"
    checksums_dir.mkdir(parents=True, exist_ok=True)
    write_sha256sums(checksums_dir / "SHA256SUMS", tuple(checksum_entries))
    return tmp_path


def test_ready_manifest_accepted(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=5)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    manifest = reader.read_manifest()
    assert manifest.status is DataPackStatus.READY
    assert manifest.record_count == 5


def test_non_ready_manifest_rejected(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=2, status=DataPackStatus.BUILDING)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    with pytest.raises(DataPackReaderIntegrityError, match="READY"):
        reader.read_manifest()


def test_malformed_manifest_rejected(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest" / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text("{not-json", encoding="utf-8")
    reader = FilesystemDataPackBootstrapReader(tmp_path)
    with pytest.raises(DataPackReaderIntegrityError):
        reader.read_manifest()


def test_streaming_pairing_matches_manifest_count(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=7, shard_size=3)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    pairs = list(reader.iter_paired_records())
    assert len(pairs) == 7
    assert [pair.relational.global_row_index for pair in pairs] == list(range(7))


def test_missing_relational_shard_fails(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=2)
    (pack_root / "relational" / "part-000001.parquet").unlink()
    reader = FilesystemDataPackBootstrapReader(pack_root)
    with pytest.raises(DataPackReaderIntegrityError, match="missing relational"):
        list(reader.iter_paired_records())


def test_identity_mismatch_fails(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=1)
    relational = _build_relational(0, "0")
    bad_embedding = _build_embedding(_build_relational(0, "1"))
    embedding_path = pack_root / "embeddings" / "part-000001.parquet"
    write_embedding_parquet(embedding_path, (bad_embedding,), embedding_dimension=_EMBEDDING_DIMENSION)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    with pytest.raises(DataPackReaderIntegrityError, match="source_ref mismatch"):
        list(reader.iter_paired_records())


def test_ordering_violation_fails(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=2)
    relational_a = _build_relational(0, "0")
    relational_b = _build_relational(1, "1")
    swapped = (relational_b, relational_a)
    relational_path = pack_root / "relational" / "part-000001.parquet"
    write_relational_parquet(relational_path, swapped)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    with pytest.raises(DataPackReaderOrderingError, match="ORDERING_VIOLATION"):
        list(reader.iter_paired_records())


def test_close_idempotent(tmp_path: Path) -> None:
    reader = FilesystemDataPackBootstrapReader(_write_ready_pack(tmp_path, record_count=1))
    reader.close()
    reader.close()
    with pytest.raises(DataPackReaderIntegrityError, match="closed"):
        reader.read_manifest()


def test_empty_stream_batches() -> None:
    assert list(iter_record_batches([], batch_size=3)) == []


def test_final_partial_batch_streaming() -> None:
    batches = list(iter_record_batches(_build_pairs(5), batch_size=2))
    assert [len(batch) for _, batch in batches] == [2, 2, 1]


def test_gap_in_global_row_index_rejected() -> None:
    first = _build_pairs(1)[0]
    third = _build_pairs(3)[2]
    with pytest.raises(StorageBootstrapIdentityError, match="gap"):
        list(iter_record_batches((first, third), batch_size=2))


@dataclass
class _InstrumentedReader:
    manifest: DataPackManifest
    total_records: int
    max_generated_index: int = -1
    iter_called: bool = False

    def read_manifest(self) -> DataPackManifest:
        return self.manifest

    def iter_paired_records(self) -> Iterator[object]:
        self.iter_called = True
        for index in range(self.total_records):
            self.max_generated_index = max(self.max_generated_index, index)
            relational = _build_relational(index, str(index))
            from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
                PairedDataPackRecord,
            )

            yield PairedDataPackRecord(relational=relational, embedding=_build_embedding(relational))

    def close(self) -> None:
        return None


def test_plan_only_does_not_iterate_records() -> None:
    reader = _InstrumentedReader(manifest=_ready_manifest(100), total_records=100)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=FakeRelationalAdapter(),
            vector=FakeVectorAdapter(),
            checkpoint_store=InMemoryBootstrapCheckpointStore(),
        )
    )
    result = service.plan(
        BootstrapRequest(
            artifact_root=Path("/tmp/pack"),
            relational_target=RelationalTargetId("r"),
            vector_target=VectorTargetId("v"),
            batch_size=BootstrapBatchSize(10),
            plan_only=True,
        )
    )
    assert result.plan.record_count == 100
    assert reader.iter_called is False


@dataclass
class _LookaheadReader:
    manifest: DataPackManifest
    total_records: int
    batch_size: int
    max_generated_index: int = -1
    write_calls: int = 0

    def read_manifest(self) -> DataPackManifest:
        return self.manifest

    def iter_paired_records(self) -> Iterator[object]:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
            PairedDataPackRecord,
        )

        for index in range(self.total_records):
            self.max_generated_index = max(self.max_generated_index, index)
            if self.write_calls == 0 and index >= self.batch_size + 2:
                raise AssertionError("reader generated too far ahead of first provider write")
            relational = _build_relational(index, str(index))
            yield PairedDataPackRecord(relational=relational, embedding=_build_embedding(relational))

    def close(self) -> None:
        return None


@dataclass
class _CountingRelationalAdapter(FakeRelationalAdapter):
    reader: _LookaheadReader | None = None

    def write_batch(self, batch):  # type: ignore[no-untyped-def]
        if self.reader is not None:
            self.reader.write_calls += 1
        return super().write_batch(batch)


def test_bounded_lookahead_on_synthetic_stream() -> None:
    reader = _LookaheadReader(
        manifest=_ready_manifest(100_000),
        total_records=100_000,
        batch_size=50,
    )
    relational = _CountingRelationalAdapter(reader=reader)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=relational,
            vector=FakeVectorAdapter(),
            checkpoint_store=InMemoryBootstrapCheckpointStore(),
        )
    )
    result = service.run(
        BootstrapRequest(
            artifact_root=Path("/tmp/pack"),
            relational_target=RelationalTargetId("r"),
            vector_target=VectorTargetId("v"),
            batch_size=BootstrapBatchSize(50),
            verification_mode=VerificationMode.SKIP,
        )
    )
    assert result.status is BootstrapFinalStatus.SUCCESS
    assert reader.write_calls > 0


def test_filesystem_reader_composition_with_fakes(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=4, shard_size=2)
    reader = FilesystemDataPackBootstrapReader(pack_root)
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=FakeRelationalAdapter(),
            vector=FakeVectorAdapter(),
            checkpoint_store=InMemoryBootstrapCheckpointStore(),
        )
    )
    result = service.run(
        BootstrapRequest(
            artifact_root=pack_root,
            relational_target=RelationalTargetId("r"),
            vector_target=VectorTargetId("v"),
            batch_size=BootstrapBatchSize(2),
            verification_mode=VerificationMode.SKIP,
        )
    )
    assert result.status is BootstrapFinalStatus.SUCCESS
    assert result.total_relational_written == 4


def test_resume_skips_provider_writes_without_materializing(tmp_path: Path) -> None:
    pack_root = _write_ready_pack(tmp_path, record_count=6, shard_size=3)
    checkpoint_store = InMemoryBootstrapCheckpointStore()
    reader = FilesystemDataPackBootstrapReader(pack_root)
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter()
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=relational,
            vector=vector,
            checkpoint_store=checkpoint_store,
        )
    )
    first = service.run(
        BootstrapRequest(
            artifact_root=pack_root,
            relational_target=RelationalTargetId("r"),
            vector_target=VectorTargetId("v"),
            batch_size=BootstrapBatchSize(2),
            verification_mode=VerificationMode.SKIP,
        )
    )
    assert first.committed_batches == 3
    second = service.run(
        BootstrapRequest(
            artifact_root=pack_root,
            relational_target=RelationalTargetId("r"),
            vector_target=VectorTargetId("v"),
            batch_size=BootstrapBatchSize(2),
            resume_mode=ResumeMode.RESUME,
            verification_mode=VerificationMode.SKIP,
        )
    )
    assert second.committed_batches == 0
    assert second.total_relational_written == 0


def test_batching_module_has_no_materialization_helpers() -> None:
    source = (_DATA_PACK_LOAD_ROOT / "batching.py").read_text(encoding="utf-8")
    assert "sorted(" not in source
    assert "list(" not in source
    assert "len(records" not in source


def test_service_has_no_full_materialization() -> None:
    source = (_DATA_PACK_LOAD_ROOT / "service.py").read_text(encoding="utf-8")
    assert "_collect_paired_records" not in source
    assert "sorted(" not in source
    assert "list(self.dependencies.reader" not in source


def test_ports_use_bootstrap_request_and_result() -> None:
    source = (_DATA_PACK_LOAD_ROOT / "ports.py").read_text(encoding="utf-8")
    assert "request: BootstrapRequest" in source
    assert "-> BootstrapResult" in source
    assert "request: object" not in source
