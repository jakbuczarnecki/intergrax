"""Temp shard write, serialized validation, and commit helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_integrity import (
    compute_source_ref_set_sha256,
    source_ref_keys_from_refs,
    validate_global_row_index_ascending,
    validate_no_duplicate_source_refs,
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
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildError,
    VpiDataPackIntegrityError,
    VpiDataPackValidationError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_key,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationStatus,
)


@dataclass(frozen=True, slots=True)
class ValidatedTempShardPair:
    relational_temp_path: Path
    embedding_temp_path: Path
    relational_relative_path: str
    embedding_relative_path: str
    relational_sha256: str
    embedding_sha256: str
    relational_source_ref_set_sha256: str
    embedding_source_ref_set_sha256: str


def write_temp_shard(
    directory: Path,
    shard_ordinal: int,
    write_callable: Callable[[Path], None],
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = temp_shard_path(directory, shard_ordinal)
    final_path = final_shard_path(directory, shard_ordinal)
    if temp_path.exists():
        temp_path.unlink()
    if final_path.exists():
        raise VpiDataPackBuildError(
            f"final shard file already exists before temp write: {final_path.name}"
        )
    write_callable(temp_path)
    if not temp_path.is_file():
        raise VpiDataPackBuildError(f"temp shard was not written: {temp_path.name}")
    return temp_path


def validate_temp_relational_shard(
    temp_path: Path,
    *,
    expected_count: int,
) -> tuple[RelationalDataPackRecord, ...]:
    try:
        records = read_relational_parquet(temp_path)
    except VpiDataPackIntegrityError as exc:
        raise VpiDataPackValidationError(str(exc)) from exc
    assert_validation_pass(
        validate_relational_records(records, expected_count=expected_count),
        stage="temp_relational_validation",
    )
    keys = source_ref_keys_from_refs(record.source_ref for record in records)
    duplicate_check = validate_no_duplicate_source_refs(
        keys,
        check_name="temp_relational_duplicate_refs",
    )
    if duplicate_check.status is not ValidationStatus.PASS:
        raise VpiDataPackValidationError(duplicate_check.detail)
    if len(records) > 1:
        ascending_check = validate_global_row_index_ascending(
            records,
            check_name="temp_relational_row_order",
        )
        if ascending_check.status is not ValidationStatus.PASS:
            raise VpiDataPackValidationError(ascending_check.detail)
    return records


def validate_temp_embedding_shard(
    temp_path: Path,
    *,
    expected_count: int,
    expected_dimension: int,
) -> tuple[EmbeddingDataPackRecord, ...]:
    try:
        records = read_embedding_parquet(temp_path, expected_dimension=expected_dimension)
    except VpiDataPackIntegrityError as exc:
        raise VpiDataPackValidationError(str(exc)) from exc
    assert_validation_pass(
        validate_embedding_records(
            records,
            expected_count=expected_count,
            expected_dimension=expected_dimension,
        ),
        stage="temp_embedding_validation",
    )
    keys = source_ref_keys_from_refs(record.source_ref for record in records)
    duplicate_check = validate_no_duplicate_source_refs(
        keys,
        check_name="temp_embedding_duplicate_refs",
    )
    if duplicate_check.status is not ValidationStatus.PASS:
        raise VpiDataPackValidationError(duplicate_check.detail)
    return records


def validate_temp_shard_pair(
    *,
    relational_temp_path: Path,
    embedding_temp_path: Path,
    expected_count: int,
    expected_dimension: int,
) -> tuple[tuple[RelationalDataPackRecord, ...], tuple[EmbeddingDataPackRecord, ...]]:
    relational_records = validate_temp_relational_shard(
        relational_temp_path,
        expected_count=expected_count,
    )
    embedding_records = validate_temp_embedding_shard(
        embedding_temp_path,
        expected_count=expected_count,
        expected_dimension=expected_dimension,
    )
    expected_refs = frozenset(source_ref_key(record.source_ref) for record in relational_records)
    assert_validation_pass(
        validate_cross_artifact_identity(
            relational_records,
            embedding_records,
            expected_refs=expected_refs,
        ),
        stage="temp_cross_ref_validation",
    )
    assert_validation_pass(
        validate_semantic_text_hashes(relational_records, embedding_records),
        stage="temp_semantic_text_hash_validation",
    )
    return relational_records, embedding_records


def prepare_validated_temp_shard_pair(
    *,
    relational_temp_path: Path,
    embedding_temp_path: Path,
    relational_relative_path: str,
    embedding_relative_path: str,
    expected_count: int,
    expected_dimension: int,
) -> ValidatedTempShardPair:
    relational_records, embedding_records = validate_temp_shard_pair(
        relational_temp_path=relational_temp_path,
        embedding_temp_path=embedding_temp_path,
        expected_count=expected_count,
        expected_dimension=expected_dimension,
    )
    relational_digest = compute_source_ref_set_sha256(
        record.source_ref for record in relational_records
    )
    embedding_digest = compute_source_ref_set_sha256(
        record.source_ref for record in embedding_records
    )
    if relational_digest != embedding_digest:
        raise VpiDataPackBuildError("temp relational/embedding source-ref digest mismatch")
    return ValidatedTempShardPair(
        relational_temp_path=relational_temp_path,
        embedding_temp_path=embedding_temp_path,
        relational_relative_path=relational_relative_path,
        embedding_relative_path=embedding_relative_path,
        relational_sha256=sha256_file(relational_temp_path),
        embedding_sha256=sha256_file(embedding_temp_path),
        relational_source_ref_set_sha256=relational_digest,
        embedding_source_ref_set_sha256=embedding_digest,
    )


def commit_temp_shard(directory: Path, shard_ordinal: int) -> Path:
    temp_path = temp_shard_path(directory, shard_ordinal)
    final_path = final_shard_path(directory, shard_ordinal)
    if not temp_path.is_file():
        raise VpiDataPackBuildError(f"temp shard missing for commit: {temp_path.name}")
    if final_path.exists():
        raise VpiDataPackBuildError(f"final shard already exists before commit: {final_path.name}")
    temp_path.replace(final_path)
    if not final_path.is_file():
        raise VpiDataPackBuildError(f"final shard missing after commit: {final_path.name}")
    return final_path


def commit_temp_shard_pair(
    *,
    relational_dir: Path,
    embeddings_dir: Path,
    shard_ordinal: int,
    embedding_commit_guard: Callable[[], None] | None = None,
) -> tuple[Path, Path]:
    relational_final = commit_temp_shard(relational_dir, shard_ordinal)
    if embedding_commit_guard is not None:
        embedding_commit_guard()
    embedding_final = commit_temp_shard(embeddings_dir, shard_ordinal)
    return relational_final, embedding_final
