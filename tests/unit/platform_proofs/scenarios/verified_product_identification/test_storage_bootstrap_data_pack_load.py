"""Unit tests for provider-neutral Data Pack storage bootstrap contracts."""

from __future__ import annotations

import ast
import json
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    compute_data_pack_content_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.embedding import (
    EmbeddingDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
    semantic_text_hash,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
    EmbeddingPackIdentity,
    SourceDatasetIdentity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.status import (
    DataPackStatus,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.batching import (
    compute_bootstrap_plan,
    iter_record_batches,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.contracts import (
    BootstrapBatchPhase,
    BootstrapBatchSize,
    BootstrapFinalStatus,
    BootstrapPlan,
    BootstrapProgress,
    BootstrapRequest,
    RelationalBatch,
    RelationalTargetId,
    ResumeMode,
    StorageLoadBatchResult,
    VectorBatch,
    VectorTargetId,
    VerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.errors import (
    BootstrapFailureCategory,
    StorageBootstrapIdentityError,
    StorageBootstrapWriteError,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.mapping import (
    assert_paired_identity,
    identity_key,
    paired_load_records,
    relational_load_record_from_pack,
    vector_load_record_from_pack,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.ports import (
    PairedDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.codec import (
    run_identity_digest,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.errors import (
    CheckpointAlreadyExists,
    CheckpointConcurrentModification,
    CheckpointNotFound,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.filesystem_store import (
    FilesystemBootstrapCheckpointStore,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.service import (
    StorageBootstrapDependencies,
    StorageBootstrapService,
)

from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.manifest.deterministic_ids import (
    search_representation_point_id,
)

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATA_PACK_LOAD_ROOT = (
    _REPO_ROOT
    / "platform_proofs/scenarios/verified_product_identification/storage_bootstrap/data_pack_load"
)
_FORBIDDEN_PROVIDER_IMPORTS = frozenset(
    {
        "psycopg",
        "asyncpg",
        "sqlalchemy",
        "mysql",
        "qdrant",
        "pgvector",
        "torch",
        "sentence_transformers",
        "transformers",
    }
)
_EMBEDDING_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
_EMBEDDING_DIMENSION = 8

pytestmark = pytest.mark.unit


@dataclass
class InMemoryBootstrapCheckpointStore:
    states: dict[str, object] = field(default_factory=dict)

    def load(self, run_identity: object) -> object | None:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
            BootstrapRunIdentity,
        )

        assert isinstance(run_identity, BootstrapRunIdentity)
        return self.states.get(run_identity_digest(run_identity))

    def initialize(self, run_identity: object, state: object) -> object:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
            BootstrapCheckpointState,
            BootstrapRunIdentity,
        )

        assert isinstance(run_identity, BootstrapRunIdentity)
        assert isinstance(state, BootstrapCheckpointState)
        digest = run_identity_digest(run_identity)
        if digest in self.states:
            raise CheckpointAlreadyExists(f"checkpoint already exists: {digest}")
        self.states[digest] = state
        return state

    def commit_batch(
        self,
        *,
        run_identity: object,
        expected_revision: int,
        checkpoint: object,
    ) -> object:
        from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.data_pack_load.checkpoint.contracts import (
            BootstrapCheckpointState,
            BootstrapRunIdentity,
        )

        assert isinstance(run_identity, BootstrapRunIdentity)
        assert isinstance(checkpoint, BootstrapCheckpointState)
        digest = run_identity_digest(run_identity)
        current = self.states.get(digest)
        if current is None:
            raise CheckpointNotFound(f"checkpoint missing: {digest}")
        assert isinstance(current, BootstrapCheckpointState)
        if current.state_revision != expected_revision:
            raise CheckpointConcurrentModification(
                f"expected revision {expected_revision}, found {current.state_revision}"
            )
        self.states[digest] = checkpoint
        return checkpoint


def _sample_json(offer_suffix: str) -> str:
    return json.dumps(
        {
            "id": f"offer-{offer_suffix}",
            "cluster_id": 1,
            "category": "Electronics",
            "identifiers": [{"key": "gtin", "value": "1234567890123"}],
            "title": f"Widget {offer_suffix}",
            "description": "Deterministic proof widget.",
            "brand": "ProofBrand",
            "price": "9.99",
            "keyValuePairs": {"Voltage": "24V"},
            "specTableContent": "| Voltage | 24V |",
        }
    )


def _build_relational(global_row_index: int, offer_suffix: str) -> RelationalDataPackRecord:
    record_json = _sample_json(offer_suffix)
    source_offer = parse_wdc_source_offer_json(record_json)
    source_ref = build_source_record_ref(source_offer, catalog_id="wdc-v2-selected")
    representation = derive_search_representation(source_offer, source_ref=source_ref)
    semantic_text = representation.semantic.semantic_text
    return RelationalDataPackRecord(
        global_row_index=global_row_index,
        source_ref=source_ref,
        record_json=record_json,
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


def _build_embedding(
    relational: RelationalDataPackRecord,
    *,
    dimension: int = _EMBEDDING_DIMENSION,
) -> EmbeddingDataPackRecord:
    vector = tuple(0.01 * index for index in range(dimension))
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
        embedding_model_revision=_EMBEDDING_REVISION,
        embedding_dimension=dimension,
        dense_embedding=vector,
    )


def _build_pairs(count: int) -> tuple[PairedDataPackRecord, ...]:
    pairs: list[PairedDataPackRecord] = []
    for index in range(count):
        relational = _build_relational(index, str(index))
        pairs.append(
            PairedDataPackRecord(
                relational=relational,
                embedding=_build_embedding(relational),
            )
        )
    return tuple(pairs)


def _ready_manifest(record_count: int) -> DataPackManifest:
    source_dataset = SourceDatasetIdentity(
        dataset_name="offers",
        dataset_path="/tmp/selected_offers.parquet",
        dataset_sha256="abc",
        dataset_record_count=record_count,
    )
    embedding_identity = EmbeddingPackIdentity(
        provider="hf",
        model="BAAI/bge-m3",
        model_revision=_EMBEDDING_REVISION,
        artifact_fingerprint=None,
        dimension=_EMBEDDING_DIMENSION,
        embedding_configuration_version="v1",
        input_policy_version="v2",
    )
    content_identity = compute_data_pack_content_identity(
        source_dataset=source_dataset,
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
    )
    return DataPackManifest(
        data_pack_version="vpi.data_pack/1.0.0",
        content_identity=content_identity,
        scenario_id="verified_product_identification",
        source_dataset=source_dataset,
        source_record_count=record_count,
        sample_identity=None,
        derivation_version="v2",
        semantic_text_version="v2",
        embedding_identity=embedding_identity,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        relational_format="parquet",
        embedding_format="parquet",
        shard_count=1,
        record_count=record_count,
        created_at_utc="2026-09-06T00:00:00+00:00",
        status=DataPackStatus.READY,
        checksums_path="checksums/SHA256SUMS",
        shards_index_path="indexes/shards.json",
        build_execution_provenance=None,
    )


@dataclass
class FakeDataPackReader:
    manifest: DataPackManifest
    pairs: tuple[PairedDataPackRecord, ...]

    def read_manifest(self) -> DataPackManifest:
        return self.manifest

    def iter_paired_records(self) -> Iterator[PairedDataPackRecord]:
        yield from self.pairs

    def close(self) -> None:
        return None


def _success_result(requested: int) -> StorageLoadBatchResult:
    return StorageLoadBatchResult(
        requested_count=requested,
        written_count=requested,
        updated_count=0,
        skipped_count=0,
        failed_count=0,
    )


@dataclass
class FakeRelationalAdapter:
    storage: dict[str, object] = field(default_factory=dict)
    fail_on_batch: int | None = None
    partial_on_batch: int | None = None
    verify_mismatch_on_batch: int | None = None
    adapter_label: str = "primary"

    def write_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult:
        if self.fail_on_batch == batch.batch_number:
            raise StorageBootstrapWriteError(f"{self.adapter_label} relational write failed")
        if self.partial_on_batch == batch.batch_number:
            return StorageLoadBatchResult(
                requested_count=len(batch.records),
                written_count=len(batch.records) - 1,
                updated_count=0,
                skipped_count=0,
                failed_count=1,
                first_failed_identity=identity_key(batch.records[-1].source_ref),
            )
        written = 0
        skipped = 0
        for record in batch.records:
            key = identity_key(record.source_ref)
            if key in self.storage:
                skipped += 1
            else:
                self.storage[key] = record
                written += 1
        return StorageLoadBatchResult(
            requested_count=len(batch.records),
            written_count=written,
            updated_count=0,
            skipped_count=skipped,
            failed_count=0,
        )

    def verify_batch(self, batch: RelationalBatch) -> StorageLoadBatchResult:
        if self.verify_mismatch_on_batch == batch.batch_number:
            return StorageLoadBatchResult(
                requested_count=len(batch.records),
                written_count=len(batch.records) - 1,
                updated_count=0,
                skipped_count=0,
                failed_count=1,
                first_failed_identity=identity_key(batch.records[0].source_ref),
            )
        present = sum(
            1 for record in batch.records if identity_key(record.source_ref) in self.storage
        )
        return StorageLoadBatchResult(
            requested_count=len(batch.records),
            written_count=present,
            updated_count=0,
            skipped_count=0,
            failed_count=len(batch.records) - present,
            first_failed_identity=None
            if present == len(batch.records)
            else identity_key(batch.records[0].source_ref),
        )


@dataclass
class FakeVectorAdapter:
    storage: dict[str, object] = field(default_factory=dict)
    fail_on_batch: int | None = None
    reject_dimension: int | None = None
    verify_mismatch_on_batch: int | None = None
    adapter_label: str = "primary"

    def write_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        if self.fail_on_batch == batch.batch_number:
            raise StorageBootstrapWriteError(f"{self.adapter_label} vector write failed")
        for record in batch.records:
            if self.reject_dimension is not None and record.embedding_dimension != self.reject_dimension:
                return StorageLoadBatchResult(
                    requested_count=len(batch.records),
                    written_count=0,
                    updated_count=0,
                    skipped_count=0,
                    failed_count=len(batch.records),
                    first_failed_identity=identity_key(record.source_ref),
                )
        written = 0
        skipped = 0
        for record in batch.records:
            key = identity_key(record.source_ref)
            if key in self.storage:
                skipped += 1
            else:
                self.storage[key] = record
                written += 1
        return StorageLoadBatchResult(
            requested_count=len(batch.records),
            written_count=written,
            updated_count=0,
            skipped_count=skipped,
            failed_count=0,
        )

    def verify_batch(self, batch: VectorBatch) -> StorageLoadBatchResult:
        if self.verify_mismatch_on_batch == batch.batch_number:
            return StorageLoadBatchResult(
                requested_count=len(batch.records),
                written_count=0,
                updated_count=0,
                skipped_count=0,
                failed_count=len(batch.records),
                first_failed_identity=identity_key(batch.records[0].source_ref),
            )
        present = sum(
            1 for record in batch.records if identity_key(record.source_ref) in self.storage
        )
        return StorageLoadBatchResult(
            requested_count=len(batch.records),
            written_count=present,
            updated_count=0,
            skipped_count=0,
            failed_count=len(batch.records) - present,
        )


def _request(
    *,
    batch_size: int = 3,
    plan_only: bool = False,
    verification_mode: VerificationMode = VerificationMode.STRICT,
    resume_mode: ResumeMode = ResumeMode.FRESH,
) -> BootstrapRequest:
    return BootstrapRequest(
        artifact_root=Path("/tmp/vpi-fixture-pack"),
        relational_target=RelationalTargetId("vpi-products"),
        vector_target=VectorTargetId("vpi-product-embeddings"),
        batch_size=BootstrapBatchSize(batch_size),
        resume_mode=resume_mode,
        verification_mode=verification_mode,
        plan_only=plan_only,
    )


def _service(
    pair_count: int,
    *,
    relational: FakeRelationalAdapter | None = None,
    vector: FakeVectorAdapter | None = None,
    status: DataPackStatus = DataPackStatus.READY,
    checkpoint_store: InMemoryBootstrapCheckpointStore | None = None,
) -> StorageBootstrapService:
    pairs = _build_pairs(pair_count)
    reader = FakeDataPackReader(
        manifest=replace(_ready_manifest(pair_count), status=status),
        pairs=pairs,
    )
    return StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=relational or FakeRelationalAdapter(),
            vector=vector or FakeVectorAdapter(),
            checkpoint_store=checkpoint_store or InMemoryBootstrapCheckpointStore(),
        )
    )


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module.split(".")[0])
    return imports


# --- CONTRACTS ---


def test_valid_bootstrap_request() -> None:
    request = _request()
    assert request.relational_target == RelationalTargetId("vpi-products")
    assert request.batch_size.value == 3


def test_invalid_batch_size_rejected() -> None:
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        BootstrapBatchSize(0)


def test_logical_targets_typed() -> None:
    assert isinstance(RelationalTargetId("x"), str)
    assert isinstance(VectorTargetId("y"), str)


def test_immutable_contracts() -> None:
    plan = BootstrapPlan(
        record_count=10,
        batch_size=3,
        batch_count=4,
        final_batch_size=1,
        relational_target=RelationalTargetId("r"),
        vector_target=VectorTargetId("v"),
    )
    with pytest.raises(AttributeError):
        plan.record_count = 5  # type: ignore[misc]


# --- STREAMING / BATCHING ---


def test_deterministic_batch_boundaries() -> None:
    pairs = _build_pairs(7)
    batches = list(iter_record_batches(pairs, batch_size=3))
    assert [len(batch) for _, batch in batches] == [3, 3, 1]
    assert batches[0][0] == 0
    assert batches[2][0] == 2


def test_bounded_processing_plan() -> None:
    plan = compute_bootstrap_plan(
        record_count=7,
        batch_size=3,
        relational_target=RelationalTargetId("r"),
        vector_target=VectorTargetId("v"),
    )
    assert plan.batch_count == 3
    assert plan.final_batch_size == 1


def test_final_partial_batch_handled() -> None:
    result = _service(5).run(_request(batch_size=2))
    assert result.committed_batches == 3
    assert result.status is BootstrapFinalStatus.SUCCESS


def test_canonical_ordering_preserved() -> None:
    pairs = list(_build_pairs(4))
    pairs.reverse()
    batches = list(iter_record_batches(pairs, batch_size=2))
    first_batch_indices = [pair.relational.global_row_index for pair in batches[0][1]]
    assert first_batch_indices == [0, 1]


# --- IDENTITY ---


def test_source_ref_parity_pass() -> None:
    pair = _build_pairs(1)[0]
    assert_paired_identity(pair)


def test_source_ref_mismatch_fail() -> None:
    relational = _build_relational(0, "0")
    embedding = _build_embedding(relational)
    mismatched = EmbeddingDataPackRecord(
        logical_point_id=embedding.logical_point_id,
        source_ref=SourceRecordRef(
            offer_id=ProductOfferId("other-offer"),
            catalog_id=relational.source_ref.catalog_id,
        ),
        derivation_version=embedding.derivation_version,
        semantic_text_hash=embedding.semantic_text_hash,
        embedding_provider=embedding.embedding_provider,
        embedding_model=embedding.embedding_model,
        embedding_model_revision=embedding.embedding_model_revision,
        embedding_dimension=embedding.embedding_dimension,
        dense_embedding=embedding.dense_embedding,
    )
    with pytest.raises(StorageBootstrapIdentityError):
        assert_paired_identity(PairedDataPackRecord(relational=relational, embedding=mismatched))


def test_logical_point_identity_preserved() -> None:
    pair = _build_pairs(1)[0]
    vector_record = vector_load_record_from_pack(pair.embedding)
    assert vector_record.logical_point_id == pair.embedding.logical_point_id


# --- RELATIONAL PORT ---


def test_successful_relational_write() -> None:
    relational = FakeRelationalAdapter()
    batch = RelationalBatch(
        batch_number=0,
        target=RelationalTargetId("r"),
        records=(relational_load_record_from_pack(_build_relational(0, "0")),),
    )
    result = relational.write_batch(batch)
    assert result.written_count == 1


def test_idempotent_retry() -> None:
    relational = FakeRelationalAdapter()
    batch = RelationalBatch(
        batch_number=0,
        target=RelationalTargetId("r"),
        records=(relational_load_record_from_pack(_build_relational(0, "0")),),
    )
    first = relational.write_batch(batch)
    second = relational.write_batch(batch)
    assert first.written_count == 1
    assert second.skipped_count == 1


def test_partial_relational_write_rejected() -> None:
    result = _service(4, relational=FakeRelationalAdapter(partial_on_batch=1)).run(_request(batch_size=2))
    assert result.status is BootstrapFinalStatus.PARTIAL
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.RELATIONAL_WRITE_FAILED


# --- VECTOR PORT ---


def test_successful_vector_write() -> None:
    relational = _build_relational(0, "0")
    vector = FakeVectorAdapter()
    batch = VectorBatch(
        batch_number=0,
        target=VectorTargetId("v"),
        records=(vector_load_record_from_pack(_build_embedding(relational)),),
    )
    result = vector.write_batch(batch)
    assert result.written_count == 1


def test_no_embedding_generation_in_core() -> None:
    source = (_DATA_PACK_LOAD_ROOT / "service.py").read_text(encoding="utf-8")
    forbidden = ("embed_batch", "sentence_transformers", "torch", "transformers", "cuda")
    assert all(token not in source.lower() for token in forbidden)


def test_wrong_vector_dimension_rejected() -> None:
    relational = _build_relational(0, "0")
    bad_embedding = _build_embedding(relational, dimension=4)
    vector = FakeVectorAdapter(reject_dimension=_EMBEDDING_DIMENSION)
    batch = VectorBatch(
        batch_number=0,
        target=VectorTargetId("v"),
        records=(vector_load_record_from_pack(bad_embedding),),
    )
    result = vector.write_batch(batch)
    assert result.failed_count == 1


# --- COORDINATION ---


def test_relational_and_vector_success_committed() -> None:
    result = _service(4).run(_request(batch_size=2))
    assert result.committed_batches == 2
    assert result.total_relational_written == 4
    assert result.total_vectors_written == 4


def test_relational_fail_batch_not_committed() -> None:
    result = _service(4, relational=FakeRelationalAdapter(fail_on_batch=0)).run(_request(batch_size=2))
    assert result.committed_batches == 0
    assert result.status is BootstrapFinalStatus.FAILED


def test_vector_fail_after_relational_not_committed() -> None:
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter(fail_on_batch=1)
    result = _service(6, relational=relational, vector=vector).run(_request(batch_size=2))
    assert result.committed_batches == 1
    assert result.status is BootstrapFinalStatus.PARTIAL
    assert len(relational.storage) == 4


def test_retry_after_vector_failure_no_relational_duplicate() -> None:
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter(fail_on_batch=1)
    checkpoint_store = InMemoryBootstrapCheckpointStore()
    first = _service(
        4,
        relational=relational,
        vector=vector,
        checkpoint_store=checkpoint_store,
    ).run(_request(batch_size=2))
    vector.fail_on_batch = None
    second = _service(
        4,
        relational=relational,
        vector=vector,
        checkpoint_store=checkpoint_store,
    ).run(_request(batch_size=2, resume_mode=ResumeMode.RESUME))
    assert first.committed_batches == 1
    assert second.committed_batches == 1
    assert len(relational.storage) == 4


def test_final_result_partial_failed_correct() -> None:
    failed = _service(3, vector=FakeVectorAdapter(fail_on_batch=0)).run(_request(batch_size=1))
    assert failed.status is BootstrapFinalStatus.FAILED
    partial = _service(4, vector=FakeVectorAdapter(fail_on_batch=1)).run(_request(batch_size=2))
    assert partial.status is BootstrapFinalStatus.PARTIAL


# --- VERIFICATION ---


def test_count_mismatch_rejected() -> None:
    result = _service(
        4,
        relational=FakeRelationalAdapter(verify_mismatch_on_batch=0),
    ).run(_request(batch_size=2))
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.INTEGRITY_FAILED


def test_orphan_identity_detected() -> None:
    relational = FakeRelationalAdapter()
    vector = FakeVectorAdapter()
    service = _service(2, relational=relational, vector=vector)
    result = service.run(_request(batch_size=2))
    relational.storage.pop(next(iter(relational.storage)))
    verify = relational.verify_batch(
        RelationalBatch(
            batch_number=0,
            target=RelationalTargetId("r"),
            records=tuple(
                paired_load_records(pair)[0] for pair in _build_pairs(2)
            ),
        )
    )
    assert verify.failed_count > 0
    assert result.status is BootstrapFinalStatus.SUCCESS


def test_verification_mismatch_rejected() -> None:
    result = _service(
        2,
        vector=FakeVectorAdapter(verify_mismatch_on_batch=0),
    ).run(_request(batch_size=2))
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.INTEGRITY_FAILED


# --- PORTABILITY ---


@pytest.mark.parametrize("provider", sorted(_FORBIDDEN_PROVIDER_IMPORTS))
def test_application_layer_has_no_provider_import(provider: str) -> None:
    violations: list[str] = []
    for module_path in sorted(_DATA_PACK_LOAD_ROOT.rglob("*.py")):
        if provider in _module_imports(module_path):
            violations.append(str(module_path.relative_to(_REPO_ROOT)))
    assert violations == []


# --- COMPOSITION ---


def test_same_service_with_alternate_relational_adapters() -> None:
    pairs = _build_pairs(2)
    reader = FakeDataPackReader(manifest=_ready_manifest(2), pairs=pairs)
    for label in ("adapter-a", "adapter-b"):
        service = StorageBootstrapService(
            dependencies=StorageBootstrapDependencies(
                reader=reader,
                relational=FakeRelationalAdapter(adapter_label=label),
                vector=FakeVectorAdapter(),
                checkpoint_store=InMemoryBootstrapCheckpointStore(),
            )
        )
        result = service.run(_request(batch_size=2))
        assert result.status is BootstrapFinalStatus.SUCCESS


def test_same_service_with_alternate_vector_adapters() -> None:
    pairs = _build_pairs(2)
    reader = FakeDataPackReader(manifest=_ready_manifest(2), pairs=pairs)
    for label in ("vector-a", "vector-b"):
        service = StorageBootstrapService(
            dependencies=StorageBootstrapDependencies(
                reader=reader,
                relational=FakeRelationalAdapter(),
                vector=FakeVectorAdapter(adapter_label=label),
                checkpoint_store=InMemoryBootstrapCheckpointStore(),
            )
        )
        result = service.run(_request(batch_size=2))
        assert result.status is BootstrapFinalStatus.SUCCESS


# --- PLAN / PROGRESS / PRECONDITION ---


def test_plan_only_mode() -> None:
    result = _service(9).plan(_request(batch_size=4))
    assert result.plan.batch_count == 3
    assert result.total_relational_written == 0
    assert result.status is BootstrapFinalStatus.SUCCESS


def test_non_ready_manifest_rejected() -> None:
    result = _service(2, status=DataPackStatus.BUILDING).run(_request())
    assert result.status is BootstrapFinalStatus.FAILED
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.PRECONDITION_FAILED


def test_progress_sink_receives_batch_phases() -> None:
    events: list[BootstrapProgress] = []

    class _Sink:
        def emit(self, progress: BootstrapProgress) -> None:
            events.append(progress)

    _service(2).run(_request(batch_size=2), progress_sink=_Sink())
    phases = {event.phase for event in events}
    assert BootstrapBatchPhase.COMMITTED in phases
    assert BootstrapBatchPhase.RELATIONAL_WRITING in phases


def test_identity_mismatch_in_service_fails() -> None:
    relational = _build_relational(0, "0")
    embedding = _build_embedding(relational)
    bad_pair = PairedDataPackRecord(
        relational=relational,
        embedding=EmbeddingDataPackRecord(
            logical_point_id=embedding.logical_point_id,
            source_ref=SourceRecordRef(
                offer_id=ProductOfferId("mismatch"),
                catalog_id=relational.source_ref.catalog_id,
            ),
            derivation_version=embedding.derivation_version,
            semantic_text_hash=embedding.semantic_text_hash,
            embedding_provider=embedding.embedding_provider,
            embedding_model=embedding.embedding_model,
            embedding_model_revision=embedding.embedding_model_revision,
            embedding_dimension=embedding.embedding_dimension,
            dense_embedding=embedding.dense_embedding,
        ),
    )
    reader = FakeDataPackReader(manifest=_ready_manifest(1), pairs=(bad_pair,))
    service = StorageBootstrapService(
        dependencies=StorageBootstrapDependencies(
            reader=reader,
            relational=FakeRelationalAdapter(),
            vector=FakeVectorAdapter(),
            checkpoint_store=InMemoryBootstrapCheckpointStore(),
        )
    )
    result = service.run(_request(batch_size=1))
    assert result.failure is not None
    assert result.failure.category is BootstrapFailureCategory.IDENTITY_MISMATCH
