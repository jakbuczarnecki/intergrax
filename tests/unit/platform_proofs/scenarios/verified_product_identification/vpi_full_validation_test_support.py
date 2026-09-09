"""Test support for full Data Pack validation fixtures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.resumable_builder import (
    DataPackBuildConfig,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    read_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.evidence import (
    DataPackProofReport,
    ValidationSectionResult,
    write_proof_report,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    read_manifest_file,
    write_manifest_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
    resolve_data_pack_paths,
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.validation.plan import (
    DataPackValidationExpectations,
    fixture_validation_expectations,
)
from tests.unit.platform_proofs.scenarios.verified_product_identification.vpi_resumable_builder_test_support import (
    canonical_fake_document_embedding_input_policy,
    patch_canonical_model_identity,
    run_resumable_data_pack_build_with_fake_policy,
    write_tiny_selected_dataset,
)

PASS_RECORD_COUNT = 77
PASS_SHARD_SIZE = 50


@dataclass(frozen=True, slots=True)
class ValidationFixture:
    artifact_root: Path
    expectations: DataPackValidationExpectations


class CorruptionKind(StrEnum):
    MISSING_RELATIONAL_SHARD = "missing_relational_shard"
    MISSING_EMBEDDING_SHARD = "missing_embedding_shard"
    RELATIONAL_RECORD_COUNT_MISMATCH = "relational_record_count_mismatch"
    EMBEDDING_RECORD_COUNT_MISMATCH = "embedding_record_count_mismatch"
    EMBEDDING_DIMENSION_MISMATCH = "embedding_dimension_mismatch"
    NAN_VECTOR = "nan_vector"
    INF_VECTOR = "inf_vector"
    ZERO_VECTOR = "zero_vector"
    SOURCE_REF_MISMATCH = "source_ref_mismatch"
    SEMANTIC_TEXT_HASH_MISMATCH = "semantic_text_hash_mismatch"
    DUPLICATE_GLOBAL_ROW_INDEX = "duplicate_global_row_index"
    DUPLICATE_SOURCE_REF = "duplicate_source_ref"
    DUPLICATE_LOGICAL_POINT_ID = "duplicate_logical_point_id"
    SHARD_RANGE_GAP = "shard_range_gap"
    SHARD_RANGE_OVERLAP = "shard_range_overlap"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    BUILD_STATE_NON_READY = "build_state_non_ready"
    MANIFEST_RECORD_COUNT_MISMATCH = "manifest_record_count_mismatch"
    WRONG_EMBEDDING_PROVIDER = "wrong_embedding_provider"
    WRONG_EMBEDDING_MODEL = "wrong_embedding_model"
    WRONG_MODEL_REVISION = "wrong_model_revision"
    WRONG_POLICY_VERSION = "wrong_policy_version"
    FINAL_SHARD_COUNT_MISMATCH = "final_shard_count_mismatch"
    DUPLICATE_SHARD_DESCRIPTOR = "duplicate_shard_descriptor"
    ORPHAN_CHECKSUM_ENTRY = "orphan_checksum_entry"


def _pass_section() -> ValidationSectionResult:
    return ValidationSectionResult(status="PASS", detail="fixture")


def write_minimal_proof_report(artifact_root: Path, *, record_count: int, content_identity: str) -> None:
    paths = resolve_data_pack_paths(artifact_root)
    report = DataPackProofReport(
        status=DataPackStatus.READY,
        data_pack_identity=content_identity,
        record_count=record_count,
        relational_validation=_pass_section(),
        embedding_validation=_pass_section(),
        cross_ref_validation=_pass_section(),
        checksum_validation=_pass_section(),
        semantic_text_hash_validation=_pass_section(),
        relational_load=_pass_section(),
        vector_load=_pass_section(),
        zero_embedding_calls=_pass_section(),
        retrieval_metrics=(),
        mapping_validation=_pass_section(),
        negative_match_validation=_pass_section(),
        provider_configuration="fixture",
        idempotent_reload=_pass_section(),
        warnings=(),
        known_gaps=(),
    )
    write_proof_report(paths.proof_report_file, report)


def build_valid_validation_fixture(tmp_path: Path, monkeypatch) -> ValidationFixture:
    patch_canonical_model_identity(monkeypatch)
    dataset_path, manifest_path = write_tiny_selected_dataset(
        tmp_path / "dataset",
        row_count=PASS_RECORD_COUNT,
    )
    output_root = tmp_path / "artifact"
    report = run_resumable_data_pack_build_with_fake_policy(
        DataPackBuildConfig(
            output_root=output_root,
            dataset_path=dataset_path,
            dataset_manifest_path=manifest_path,
            shard_size=PASS_SHARD_SIZE,
            max_records=PASS_RECORD_COUNT,
            start_fresh=True,
        ),
        document_embedding_input_policy=canonical_fake_document_embedding_input_policy(),
    )
    assert report.finalized
    paths = resolve_data_pack_paths(output_root)
    manifest = read_manifest_file(paths.manifest_file)
    write_minimal_proof_report(output_root, record_count=PASS_RECORD_COUNT, content_identity=manifest.content_identity)
    return ValidationFixture(
        artifact_root=output_root,
        expectations=fixture_validation_expectations(
            record_count=PASS_RECORD_COUNT,
            shard_size=PASS_SHARD_SIZE,
            require_proof_report=True,
        ),
    )


def apply_corruption(fixture: ValidationFixture, kind: CorruptionKind) -> None:
    paths = resolve_data_pack_paths(fixture.artifact_root)
    if kind is CorruptionKind.MISSING_RELATIONAL_SHARD:
        final_shard_path(paths.relational_dir, 2).unlink(missing_ok=True)
        return
    if kind is CorruptionKind.MISSING_EMBEDDING_SHARD:
        final_shard_path(paths.embeddings_dir, 2).unlink(missing_ok=True)
        return
    if kind is CorruptionKind.RELATIONAL_RECORD_COUNT_MISMATCH:
        shard_path = final_shard_path(paths.relational_dir, 1)
        records = list(read_relational_parquet(shard_path))
        write_relational_parquet(shard_path, tuple(records[:-1]))
        return
    if kind is CorruptionKind.EMBEDDING_RECORD_COUNT_MISMATCH:
        shard_path = final_shard_path(paths.embeddings_dir, 1)
        records = list(read_embedding_parquet(shard_path, expected_dimension=1024))
        write_embedding_parquet(shard_path, tuple(records[:-1]), embedding_dimension=1024)
        return
    if kind is CorruptionKind.EMBEDDING_DIMENSION_MISMATCH:
        shard_path = final_shard_path(paths.embeddings_dir, 1)
        table = pq.read_table(shard_path)
        dimensions = table.column("embedding_dimension").to_pylist()
        dimensions[0] = 512
        column_index = table.schema.get_field_index("embedding_dimension")
        corrupted_table = table.set_column(
            column_index,
            "embedding_dimension",
            pa.array(dimensions, type=pa.int32()),
        )
        pq.write_table(corrupted_table, shard_path)
        return
    if kind is CorruptionKind.NAN_VECTOR:
        _mutate_embedding_vector(paths, value=float("nan"))
        return
    if kind is CorruptionKind.INF_VECTOR:
        _mutate_embedding_vector(paths, value=float("inf"))
        return
    if kind is CorruptionKind.ZERO_VECTOR:
        _mutate_embedding_vector(paths, value=0.0)
        return
    if kind is CorruptionKind.SOURCE_REF_MISMATCH:
        shard_path = final_shard_path(paths.embeddings_dir, 1)
        records = list(read_embedding_parquet(shard_path, expected_dimension=1024))
        relational = read_relational_parquet(final_shard_path(paths.relational_dir, 1))
        replacement = records[0].__class__(
            logical_point_id=records[0].logical_point_id,
            source_ref=relational[1].source_ref,
            derivation_version=records[0].derivation_version,
            semantic_text_hash=records[0].semantic_text_hash,
            embedding_provider=records[0].embedding_provider,
            embedding_model=records[0].embedding_model,
            embedding_model_revision=records[0].embedding_model_revision,
            embedding_dimension=records[0].embedding_dimension,
            dense_embedding=records[0].dense_embedding,
        )
        records[0] = replacement
        write_embedding_parquet(shard_path, tuple(records), embedding_dimension=1024)
        return
    if kind is CorruptionKind.SEMANTIC_TEXT_HASH_MISMATCH:
        shard_path = final_shard_path(paths.embeddings_dir, 1)
        records = list(read_embedding_parquet(shard_path, expected_dimension=1024))
        corrupted = records[0]
        records[0] = corrupted.__class__(
            logical_point_id=corrupted.logical_point_id,
            source_ref=corrupted.source_ref,
            derivation_version=corrupted.derivation_version,
            semantic_text_hash="deadbeef" * 8,
            embedding_provider=corrupted.embedding_provider,
            embedding_model=corrupted.embedding_model,
            embedding_model_revision=corrupted.embedding_model_revision,
            embedding_dimension=corrupted.embedding_dimension,
            dense_embedding=corrupted.dense_embedding,
        )
        write_embedding_parquet(shard_path, tuple(records), embedding_dimension=1024)
        return
    if kind is CorruptionKind.DUPLICATE_GLOBAL_ROW_INDEX:
        shard_path = final_shard_path(paths.relational_dir, 1)
        records = list(read_relational_parquet(shard_path))
        duplicate = records[0].__class__(
            global_row_index=records[1].global_row_index,
            source_ref=records[0].source_ref,
            record_json=records[0].record_json,
            derivation_version=records[0].derivation_version,
            semantic_text=records[0].semantic_text,
            semantic_text_hash=records[0].semantic_text_hash,
            title=records[0].title,
            brand=records[0].brand,
            category=records[0].category,
            description=records[0].description,
            has_identifiers=records[0].has_identifiers,
            has_spec_table=records[0].has_spec_table,
            has_structured_attributes=records[0].has_structured_attributes,
        )
        records.append(duplicate)
        write_relational_parquet(shard_path, tuple(records))
        return
    if kind is CorruptionKind.DUPLICATE_SOURCE_REF:
        shard_path = final_shard_path(paths.relational_dir, 2)
        records = list(read_relational_parquet(shard_path))
        duplicate = records[0].__class__(
            global_row_index=records[-1].global_row_index + 1,
            source_ref=records[0].source_ref,
            record_json=records[0].record_json,
            derivation_version=records[0].derivation_version,
            semantic_text=records[0].semantic_text,
            semantic_text_hash=records[0].semantic_text_hash,
            title=records[0].title,
            brand=records[0].brand,
            category=records[0].category,
            description=records[0].description,
            has_identifiers=records[0].has_identifiers,
            has_spec_table=records[0].has_spec_table,
            has_structured_attributes=records[0].has_structured_attributes,
        )
        records.append(duplicate)
        write_relational_parquet(shard_path, tuple(records))
        return
    if kind is CorruptionKind.DUPLICATE_LOGICAL_POINT_ID:
        shard_path = final_shard_path(paths.embeddings_dir, 2)
        records = list(read_embedding_parquet(shard_path, expected_dimension=1024))
        duplicate = records[0].__class__(
            logical_point_id=records[0].logical_point_id,
            source_ref=records[1].source_ref,
            derivation_version=records[1].derivation_version,
            semantic_text_hash=records[1].semantic_text_hash,
            embedding_provider=records[1].embedding_provider,
            embedding_model=records[1].embedding_model,
            embedding_model_revision=records[1].embedding_model_revision,
            embedding_dimension=records[1].embedding_dimension,
            dense_embedding=records[1].dense_embedding,
        )
        records.append(duplicate)
        write_embedding_parquet(shard_path, tuple(records), embedding_dimension=1024)
        return
    if kind is CorruptionKind.SHARD_RANGE_GAP:
        shard_path = final_shard_path(paths.relational_dir, 2)
        records = list(read_relational_parquet(shard_path))
        shifted = [
            record.__class__(
                global_row_index=record.global_row_index + 10,
                source_ref=record.source_ref,
                record_json=record.record_json,
                derivation_version=record.derivation_version,
                semantic_text=record.semantic_text,
                semantic_text_hash=record.semantic_text_hash,
                title=record.title,
                brand=record.brand,
                category=record.category,
                description=record.description,
                has_identifiers=record.has_identifiers,
                has_spec_table=record.has_spec_table,
                has_structured_attributes=record.has_structured_attributes,
            )
            for record in records
        ]
        write_relational_parquet(shard_path, tuple(shifted))
        return
    if kind is CorruptionKind.SHARD_RANGE_OVERLAP:
        shard_path = final_shard_path(paths.relational_dir, 2)
        records = list(read_relational_parquet(shard_path))
        shifted = [
            record.__class__(
                global_row_index=record.global_row_index - 10,
                source_ref=record.source_ref,
                record_json=record.record_json,
                derivation_version=record.derivation_version,
                semantic_text=record.semantic_text,
                semantic_text_hash=record.semantic_text_hash,
                title=record.title,
                brand=record.brand,
                category=record.category,
                description=record.description,
                has_identifiers=record.has_identifiers,
                has_spec_table=record.has_spec_table,
                has_structured_attributes=record.has_structured_attributes,
            )
            for record in records
        ]
        write_relational_parquet(shard_path, tuple(shifted))
        return
    if kind is CorruptionKind.CHECKSUM_MISMATCH:
        paths.checksums_file.write_text("f" * 64 + "  manifest/manifest.json\n", encoding="utf-8")
        return
    if kind is CorruptionKind.BUILD_STATE_NON_READY:
        payload = json.loads(paths.build_state_file.read_text(encoding="utf-8"))
        payload["shards"][0]["status"] = "PENDING"
        payload["completed_shards"] = payload["shard_count"] - 1
        paths.build_state_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    if kind is CorruptionKind.MANIFEST_RECORD_COUNT_MISMATCH:
        payload = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
        payload["record_count"] = payload["record_count"] + 1
        payload["source_record_count"] = payload["record_count"]
        paths.manifest_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    if kind is CorruptionKind.WRONG_EMBEDDING_PROVIDER:
        _mutate_manifest_embedding_identity(paths, provider="openai")
        return
    if kind is CorruptionKind.WRONG_EMBEDDING_MODEL:
        _mutate_manifest_embedding_identity(paths, model="other-model")
        return
    if kind is CorruptionKind.WRONG_MODEL_REVISION:
        _mutate_manifest_embedding_identity(paths, model_revision="badrevision")
        return
    if kind is CorruptionKind.WRONG_POLICY_VERSION:
        _mutate_manifest_embedding_identity(paths, input_policy_version="wrong-policy")
        return
    if kind is CorruptionKind.FINAL_SHARD_COUNT_MISMATCH:
        payload = json.loads(paths.manifest_file.read_text(encoding="utf-8"))
        payload["shard_count"] = payload["shard_count"] + 1
        paths.manifest_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    if kind is CorruptionKind.DUPLICATE_SHARD_DESCRIPTOR:
        payload = json.loads(paths.shards_index_file.read_text(encoding="utf-8"))
        payload["relational_shards"].append(payload["relational_shards"][0])
        paths.shards_index_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    if kind is CorruptionKind.ORPHAN_CHECKSUM_ENTRY:
        lines = paths.checksums_file.read_text(encoding="utf-8").splitlines()
        lines.append("a" * 64 + "  orphan/missing.parquet")
        paths.checksums_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    msg = f"unsupported corruption kind: {kind}"
    raise ValueError(msg)


def _mutate_embedding_vector(paths, *, value: float) -> None:
    shard_path = final_shard_path(paths.embeddings_dir, 1)
    records = list(read_embedding_parquet(shard_path, expected_dimension=1024))
    corrupted = records[0]
    vector = list(corrupted.dense_embedding)
    vector[0] = value
    records[0] = corrupted.__class__(
        logical_point_id=corrupted.logical_point_id,
        source_ref=corrupted.source_ref,
        derivation_version=corrupted.derivation_version,
        semantic_text_hash=corrupted.semantic_text_hash,
        embedding_provider=corrupted.embedding_provider,
        embedding_model=corrupted.embedding_model,
        embedding_model_revision=corrupted.embedding_model_revision,
        embedding_dimension=corrupted.embedding_dimension,
        dense_embedding=tuple(vector),
    )
    write_embedding_parquet(shard_path, tuple(records), embedding_dimension=1024)


def _mutate_manifest_embedding_identity(
    paths,
    *,
    provider: str | None = None,
    model: str | None = None,
    model_revision: str | None = None,
    input_policy_version: str | None = None,
) -> None:
    manifest = read_manifest_file(paths.manifest_file)
    embedding = manifest.embedding_identity
    replacement_embedding = embedding.__class__(
        provider=provider or embedding.provider,
        model=model or embedding.model,
        model_revision=model_revision or embedding.model_revision,
        artifact_fingerprint=embedding.artifact_fingerprint,
        dimension=embedding.dimension,
        embedding_configuration_version=embedding.embedding_configuration_version,
        input_policy_version=input_policy_version or embedding.input_policy_version,
    )
    replacement = manifest.__class__(
        data_pack_version=manifest.data_pack_version,
        content_identity=manifest.content_identity,
        scenario_id=manifest.scenario_id,
        source_dataset=manifest.source_dataset,
        source_record_count=manifest.source_record_count,
        sample_identity=manifest.sample_identity,
        derivation_version=manifest.derivation_version,
        semantic_text_version=manifest.semantic_text_version,
        embedding_identity=replacement_embedding,
        relational_schema_version=manifest.relational_schema_version,
        embedding_schema_version=manifest.embedding_schema_version,
        relational_format=manifest.relational_format,
        embedding_format=manifest.embedding_format,
        shard_count=manifest.shard_count,
        record_count=manifest.record_count,
        created_at_utc=manifest.created_at_utc,
        status=manifest.status,
        checksums_path=manifest.checksums_path,
        shards_index_path=manifest.shards_index_path,
        build_execution_provenance=manifest.build_execution_provenance,
    )
    write_manifest_file(paths.manifest_file, replacement)
