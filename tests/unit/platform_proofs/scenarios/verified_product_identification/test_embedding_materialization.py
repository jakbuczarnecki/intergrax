"""Unit tests for VPI embedding artifact materialization."""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field, fields
from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    EMBEDDING_CONFIGURATION_VERSION,
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.application.domain.search_representation import (
    SEARCH_REPRESENTATION_DERIVATION_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.artifact.record import (
    EmbeddingArtifactRecord,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.config import (
    VpiEmbeddingMaterializationConfig,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.contracts.errors import (
    ArtifactCompatibilityError,
    ArtifactIntegrityError,
    EmbeddingMaterializationProviderError,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.compatibility import (
    EmbeddingArtifactCompatibilityIdentity,
    artifact_directory_fingerprint,
    assert_manifest_compatible,
    compatibility_identity_from_manifest,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.model import (
    EMBEDDING_ARTIFACT_SCHEMA_VERSION,
    EmbeddingArtifactManifest,
    EmbeddingArtifactShardDescriptor,
    EmbeddingArtifactState,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.manifest.run_target import (
    assert_requested_target_not_below_checkpoint,
    checkpoint_meets_target,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.orchestration.orchestrator import (
    EmbeddingMaterializationOrchestrator,
    VpiEmbeddingMaterializationDependencies,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.manifest_io import (
    manifest_from_dict,
    manifest_to_dict,
    read_manifest_file,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.paths import (
    shard_file_name,
    shard_path,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.reader import (
    ParquetFilesystemArtifactReader,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.reconciliation import (
    reconcile_orphan_shards,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.shard_validation import (
    sha256_file,
    validate_record_row_alignment,
    validate_shard_descriptor_continuity,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.stores.parquet.writer import (
    ParquetFilesystemArtifactWriter,
)
from platform_proofs.scenarios.verified_product_identification.embedding_materialization.validation.vectors import (
    validate_embedding_batch_vectors,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.config import (
    DatasetVerificationMode,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationReport,
    ValidationStatus,
)
from intergrax.rag.embedding.registry.profile import EmbeddingProfile

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[5]
VPI_ROOT = REPO_ROOT / "platform_proofs" / "scenarios" / "verified_product_identification"
MATERIALIZATION_ROOT = VPI_ROOT / "embedding_materialization"
ORCHESTRATOR_PATH = MATERIALIZATION_ROOT / "orchestration" / "orchestrator.py"


def _sample_identity(**overrides: object) -> EmbeddingArtifactCompatibilityIdentity:
    base = EmbeddingArtifactCompatibilityIdentity(
        dataset_checksum="abc123",
        dataset_record_count=1000,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=8,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        catalog_id="wdc-v2-selected",
        source_revision=None,
    )
    if not overrides:
        return base
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return EmbeddingArtifactCompatibilityIdentity(**values)


def _sample_manifest(**overrides: object) -> EmbeddingArtifactManifest:
    base = EmbeddingArtifactManifest(
        state=EmbeddingArtifactState.INITIALIZING,
        artifact_schema_version=EMBEDDING_ARTIFACT_SCHEMA_VERSION,
        dataset_path="/data/selected_offers.parquet",
        dataset_checksum="abc123",
        dataset_record_count=1000,
        search_representation_derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        embedding_configuration_version=EMBEDDING_CONFIGURATION_VERSION,
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=8,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        checkpoint_shard_ordinal=None,
        checkpoint_rows_materialized=0,
        target_max_records=64,
        total_artifact_record_count=0,
        shard_count=0,
        committed_shards=(),
    )
    if not overrides:
        return base
    values = {item.name: getattr(base, item.name) for item in fields(base)}
    values.update(overrides)
    return EmbeddingArtifactManifest(**values)


def _artifact_record(row_index: int, *, dimension: int = 8) -> EmbeddingArtifactRecord:
    vector = tuple(0.1 * (index + 1) for index in range(dimension))
    return EmbeddingArtifactRecord(
        global_row_index=row_index,
        logical_point_id=f"vpi:wdc-v2-selected:offer-{row_index}:semantic:v2",
        catalog_id="wdc-v2-selected",
        offer_id=f"offer-{row_index}",
        source_revision=None,
        derivation_version=SEARCH_REPRESENTATION_DERIVATION_VERSION,
        semantic_text=f"semantic {row_index}",
        lexical_text=f"lexical {row_index}",
        embedding_provider="hf",
        embedding_model="BAAI/bge-m3",
        embedding_dimension=dimension,
        dense_embedding=vector,
    )


def _materialization_config(
    dataset_path: Path,
    *,
    max_records: int = 4,
    shard_size: int = 2,
    embedding_batch_size: int = 2,
) -> VpiEmbeddingMaterializationConfig:
    manifest_path = dataset_path.parent / "selected_offers_manifest.json"
    return VpiEmbeddingMaterializationConfig(
        dataset_path=dataset_path,
        dataset_manifest_path=manifest_path if manifest_path.is_file() else None,
        dataset_verification_mode=DatasetVerificationMode.FAST,
        catalog_id="wdc-v2-selected",
        source_revision=None,
        max_records=max_records,
        source_read_batch_size=2,
        embedding_batch_size=embedding_batch_size,
        artifact_shard_size=shard_size,
        artifact_root_dir=dataset_path.parent / "artifacts",
        embedding_configuration=VpiEmbeddingConfiguration(
            profile=EmbeddingProfile(provider="hf", model="fake-model"),
            expected_dimension=8,
        ),
    )


@dataclass
class FakeEmbeddingPort:
    dimension: int = 8
    embed_calls: int = 0
    texts_seen: list[str] = field(default_factory=list)

    def probe(self) -> EmbeddingProbeResult:
        return EmbeddingProbeResult(
            status=ValidationStatus.PASS,
            provider="fake",
            model="fake-model",
            resolved_dimension=self.dimension,
            probe_vector_count=1,
            detail="ok",
        )

    def embed_batch(self, texts: tuple[str, ...] | list[str]) -> tuple[tuple[float, ...], ...]:
        self.embed_calls += 1
        self.texts_seen.extend(texts)
        return tuple(
            tuple(0.1 * (index + 1) for index in range(self.dimension))
            for _ in texts
        )

    def close(self) -> None:
        return None


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            imports.add(node.module)
    return imports


def test_identity_rejects_same_dimension_different_model() -> None:
    manifest = _sample_manifest(embedding_model="other-model")
    expected = _sample_identity()
    with pytest.raises(ArtifactCompatibilityError, match="embedding_model"):
        assert_manifest_compatible(existing=manifest, expected=expected)


def test_directory_fingerprint_changes_with_model() -> None:
    first = artifact_directory_fingerprint(_sample_identity())
    second = artifact_directory_fingerprint(
        _sample_identity(embedding_model="other-model")
    )
    assert first != second


def test_source_revision_compatible_when_matching() -> None:
    manifest = _sample_manifest(source_revision="rev-a")
    expected = _sample_identity(source_revision="rev-a")
    assert_manifest_compatible(existing=manifest, expected=expected)


def test_source_revision_rejects_drift() -> None:
    manifest = _sample_manifest(source_revision="rev-a")
    expected = _sample_identity(source_revision="rev-b")
    with pytest.raises(ArtifactCompatibilityError, match="source_revision"):
        assert_manifest_compatible(existing=manifest, expected=expected)


def test_source_revision_none_to_value_is_incompatible() -> None:
    manifest = _sample_manifest(source_revision=None)
    expected = _sample_identity(source_revision="rev-a")
    with pytest.raises(ArtifactCompatibilityError, match="source_revision"):
        assert_manifest_compatible(existing=manifest, expected=expected)


def test_source_revision_value_to_none_is_incompatible() -> None:
    manifest = _sample_manifest(source_revision="rev-a")
    expected = _sample_identity(source_revision=None)
    with pytest.raises(ArtifactCompatibilityError, match="source_revision"):
        assert_manifest_compatible(existing=manifest, expected=expected)


def test_directory_fingerprint_changes_with_source_revision() -> None:
    first = artifact_directory_fingerprint(_sample_identity(source_revision="rev-a"))
    second = artifact_directory_fingerprint(_sample_identity(source_revision="rev-b"))
    assert first != second


def test_compatibility_identity_from_manifest_includes_source_revision() -> None:
    manifest = _sample_manifest(source_revision="rev-a")
    identity = compatibility_identity_from_manifest(manifest)
    assert identity.source_revision == "rev-a"


def test_manifest_roundtrip_preserves_source_revision(tmp_path: Path) -> None:
    manifest = _sample_manifest(source_revision="rev-a")
    manifest_path = tmp_path / "manifest.json"
    write_manifest_file(manifest_path, manifest)
    restored = read_manifest_file(manifest_path)
    assert restored.source_revision == "rev-a"
    assert manifest_to_dict(restored)["source_revision"] == "rev-a"


def _manifest_wire_payload() -> dict[str, object]:
    return dict(manifest_to_dict(_sample_manifest(source_revision="rev-a")))


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (lambda payload: payload.update({"dataset_record_count": "1000"}), "dataset_record_count"),
        (lambda payload: payload.update({"committed_shards": {}}), "committed_shards"),
        (lambda payload: payload["committed_shards"].append({"shard_ordinal": "0"}), "shard_ordinal"),
        (lambda payload: payload.pop("state"), "state"),
        (lambda payload: payload.update({"source_revision": 42}), "source_revision"),
        (lambda payload: payload.update({"embedding_dimension": -1}), "embedding_dimension"),
        (lambda payload: payload.update({"checkpoint_rows_materialized": -1}), "checkpoint_rows_materialized"),
    ],
)
def test_manifest_from_dict_rejects_malformed_payload(
    mutator: object,
    match: str,
) -> None:
    payload = _manifest_wire_payload()
    mutator(payload)
    with pytest.raises(ArtifactIntegrityError, match=match):
        manifest_from_dict(payload)


def test_shard_naming_is_deterministic() -> None:
    assert shard_file_name(0) == "part-000000.parquet"
    assert shard_file_name(42) == "part-000042.parquet"


def test_validate_vectors_rejects_nan_and_wrong_count() -> None:
    with pytest.raises(EmbeddingMaterializationProviderError, match="expected 2 vectors"):
        validate_embedding_batch_vectors(
            vectors=(tuple(0.1 for _ in range(8)),),
            expected_count=2,
            expected_dimension=8,
        )
    with pytest.raises(EmbeddingMaterializationProviderError, match="non-finite"):
        validate_embedding_batch_vectors(
            vectors=(tuple(float("nan") for _ in range(8)),),
            expected_count=1,
            expected_dimension=8,
        )


def test_row_alignment_rejects_gap() -> None:
    records = (_artifact_record(0), _artifact_record(2))
    with pytest.raises(ArtifactIntegrityError, match="gap"):
        validate_record_row_alignment(records)


def test_shard_continuity_rejects_overlap() -> None:
    shards = (
        EmbeddingArtifactShardDescriptor(
            shard_ordinal=0,
            file_name="part-000000.parquet",
            first_global_row_index=0,
            last_global_row_index=1,
            record_count=2,
            sha256_checksum="a",
        ),
        EmbeddingArtifactShardDescriptor(
            shard_ordinal=1,
            file_name="part-000001.parquet",
            first_global_row_index=1,
            last_global_row_index=3,
            record_count=3,
            sha256_checksum="b",
        ),
    )
    with pytest.raises(ArtifactIntegrityError, match="overlap"):
        validate_shard_descriptor_continuity(shards)


def test_parquet_writer_atomic_commit_and_reader_iteration(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    writer = ParquetFilesystemArtifactWriter(artifact_dir)
    manifest = _sample_manifest()
    writer.prepare(manifest)
    records = tuple(_artifact_record(index) for index in range(3))
    descriptor = writer.write_shard(0, records)
    assert shard_path(artifact_dir, 0).is_file()
    assert not temp_shard_path(artifact_dir, 0).exists()
    assert descriptor.record_count == 3
    assert descriptor.sha256_checksum == sha256_file(shard_path(artifact_dir, 0))

    committed_manifest = manifest.with_checkpoint(
        shard_ordinal=0,
        rows_materialized=3,
        committed_shards=(descriptor,),
    )
    writer.write_manifest(committed_manifest)

    reader = ParquetFilesystemArtifactReader(artifact_dir)
    read_rows = list(reader.iterate_shard_records(descriptor))
    assert [row.global_row_index for row in read_rows] == [0, 1, 2]
    assert read_rows[0].lexical_text == "lexical 0"


def test_orphan_shard_reconciliation_after_crash(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    writer = ParquetFilesystemArtifactWriter(artifact_dir)
    manifest = _sample_manifest()
    writer.prepare(manifest)
    records = tuple(_artifact_record(index) for index in range(2))
    descriptor = writer.write_shard(0, records)
    manifest = manifest.with_checkpoint(
        shard_ordinal=None,
        rows_materialized=0,
        committed_shards=(),
    )
    writer.write_manifest(manifest)

    reconciled = reconcile_orphan_shards(artifact_dir=artifact_dir, manifest=manifest)
    assert reconciled.checkpoint_rows_materialized == 2
    assert reconciled.shard_count == 1
    assert reconciled.committed_shards[0].sha256_checksum == descriptor.sha256_checksum


def test_requested_target_below_checkpoint_fails() -> None:
    with pytest.raises(ArtifactCompatibilityError, match="below existing checkpoint"):
        assert_requested_target_not_below_checkpoint(
            requested_target_rows=10,
            checkpoint_rows_materialized=20,
        )


def test_materialization_resume_and_restart_without_reembed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_path = VPI_ROOT / "dataset" / "processed" / "selected_offers.parquet"
    if not dataset_path.is_file():
        pytest.skip("WDC dataset unavailable")

    artifact_dir = tmp_path / "artifact"
    config = _materialization_config(dataset_path, max_records=4, shard_size=2)
    fake_embedding = FakeEmbeddingPort()
    orchestrator = EmbeddingMaterializationOrchestrator(
        config=config,
        dependencies=VpiEmbeddingMaterializationDependencies(
            artifact_writer=ParquetFilesystemArtifactWriter(artifact_dir),
            embedding=fake_embedding,
        ),
    )
    first_report = orchestrator.run()
    assert first_report.final_state is EmbeddingArtifactState.READY
    assert first_report.rows_materialized == 4
    first_calls = fake_embedding.embed_calls

    restart_orchestrator = EmbeddingMaterializationOrchestrator(
        config=config,
        dependencies=VpiEmbeddingMaterializationDependencies(
            artifact_writer=ParquetFilesystemArtifactWriter(artifact_dir),
            embedding=fake_embedding,
        ),
    )
    second_report = restart_orchestrator.run()
    assert second_report.final_state is EmbeddingArtifactState.READY
    assert fake_embedding.embed_calls == first_calls


def test_target_extension_embeds_only_delta(
    tmp_path: Path,
) -> None:
    dataset_path = VPI_ROOT / "dataset" / "processed" / "selected_offers.parquet"
    if not dataset_path.is_file():
        pytest.skip("WDC dataset unavailable")

    artifact_dir = tmp_path / "artifact"
    fake_embedding = FakeEmbeddingPort()
    first_config = _materialization_config(dataset_path, max_records=2, shard_size=2)
    first_orchestrator = EmbeddingMaterializationOrchestrator(
        config=first_config,
        dependencies=VpiEmbeddingMaterializationDependencies(
            artifact_writer=ParquetFilesystemArtifactWriter(artifact_dir),
            embedding=fake_embedding,
        ),
    )
    first_report = first_orchestrator.run()
    assert first_report.final_state is EmbeddingArtifactState.READY
    first_calls = fake_embedding.embed_calls
    first_texts = list(fake_embedding.texts_seen)

    second_config = _materialization_config(dataset_path, max_records=4, shard_size=2)
    second_orchestrator = EmbeddingMaterializationOrchestrator(
        config=second_config,
        dependencies=VpiEmbeddingMaterializationDependencies(
            artifact_writer=ParquetFilesystemArtifactWriter(artifact_dir),
            embedding=fake_embedding,
        ),
    )
    second_report = second_orchestrator.run()
    assert second_report.final_state is EmbeddingArtifactState.READY
    assert fake_embedding.embed_calls > first_calls
    assert fake_embedding.texts_seen[: len(first_texts)] == first_texts


def test_materialization_orchestrator_has_no_storage_vendor_imports() -> None:
    imports = _module_imports(ORCHESTRATOR_PATH)
    forbidden_prefixes = (
        "qdrant",
        "psycopg",
        "sentence_transformers",
        "torch",
        "openai",
        "ollama",
        "vllm",
    )
    violations = sorted(
        imported
        for imported in imports
        if any(
            imported == prefix or imported.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )
    assert violations == []


def test_materialization_production_tree_has_no_vendor_sdk_imports() -> None:
    forbidden_prefixes = (
        "qdrant",
        "psycopg",
        "sentence_transformers",
        "torch",
        "openai",
        "ollama",
        "vllm",
    )
    violations: list[str] = []
    for module_path in sorted(MATERIALIZATION_ROOT.rglob("*.py")):
        if "stores/parquet" not in str(module_path).replace("\\", "/"):
            pass
        for imported in _module_imports(module_path):
            if any(
                imported == prefix or imported.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            ):
                violations.append(f"{module_path.relative_to(REPO_ROOT)} -> {imported}")
    assert violations == []


def test_checkpoint_meets_target() -> None:
    assert checkpoint_meets_target(
        checkpoint_rows_materialized=100,
        requested_target_rows=100,
    )
    assert not checkpoint_meets_target(
        checkpoint_rows_materialized=50,
        requested_target_rows=100,
    )
