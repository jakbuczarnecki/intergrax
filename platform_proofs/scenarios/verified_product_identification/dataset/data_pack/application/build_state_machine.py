"""Builder-local state machine, persistence, and READY shard validation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_integrity import (
    compute_source_ref_set_sha256,
    validate_embedding_shard_source_identity,
    validate_relational_shard_source_identity,
    validate_shard_pair_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.build_state import (
    DataPackBuildState,
    DataPackShardBuildState,
    DataPackShardStatus,
    read_build_state_file,
    write_build_state_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackBuildStateError,
    VpiDataPackReadyShardCorruptionError,
    VpiDataPackResumeError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.paths import (
    final_shard_path,
    shard_file_name,
    temp_shard_path,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationStatus,
)

_ALLOWED_TRANSITIONS: dict[DataPackShardStatus, frozenset[DataPackShardStatus]] = {
    DataPackShardStatus.PENDING: frozenset({DataPackShardStatus.DERIVING}),
    DataPackShardStatus.DERIVING: frozenset({DataPackShardStatus.EMBEDDING}),
    DataPackShardStatus.EMBEDDING: frozenset({DataPackShardStatus.WRITING}),
    DataPackShardStatus.WRITING: frozenset({DataPackShardStatus.VALIDATING}),
    DataPackShardStatus.VALIDATING: frozenset({DataPackShardStatus.READY}),
    DataPackShardStatus.READY: frozenset(),
}


def validate_shard_status_transition(
    current: DataPackShardStatus,
    target: DataPackShardStatus,
) -> None:
    allowed = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target not in allowed:
        raise VpiDataPackBuildStateError(
            f"invalid shard status transition: {current.value} -> {target.value}"
        )


def reset_shard_to_pending(shard: DataPackShardBuildState) -> DataPackShardBuildState:
    if shard.status is DataPackShardStatus.READY:
        raise VpiDataPackBuildStateError("READY shard cannot transition to PENDING")
    return DataPackShardBuildState(
        ordinal=shard.ordinal,
        start_row_index=shard.start_row_index,
        end_row_index_exclusive=shard.end_row_index_exclusive,
        expected_record_count=shard.expected_record_count,
        status=DataPackShardStatus.PENDING,
        relational_relative_path=None,
        embedding_relative_path=None,
        attempt=shard.attempt + 1,
        relational_sha256=None,
        embedding_sha256=None,
        relational_source_ref_set_sha256=None,
        embedding_source_ref_set_sha256=None,
        last_error_code=None,
        last_error_message=None,
    )


def transition_shard(
    shard: DataPackShardBuildState,
    target: DataPackShardStatus,
    **updates: object,
) -> DataPackShardBuildState:
    validate_shard_status_transition(shard.status, target)
    payload = {
        "ordinal": shard.ordinal,
        "start_row_index": shard.start_row_index,
        "end_row_index_exclusive": shard.end_row_index_exclusive,
        "expected_record_count": shard.expected_record_count,
        "status": target,
        "relational_relative_path": shard.relational_relative_path,
        "embedding_relative_path": shard.embedding_relative_path,
        "attempt": shard.attempt,
        "relational_sha256": shard.relational_sha256,
        "embedding_sha256": shard.embedding_sha256,
        "relational_source_ref_set_sha256": shard.relational_source_ref_set_sha256,
        "embedding_source_ref_set_sha256": shard.embedding_source_ref_set_sha256,
        "last_error_code": shard.last_error_code,
        "last_error_message": shard.last_error_message,
    }
    payload.update(updates)
    return DataPackShardBuildState(**payload)


def replace_shard(state: DataPackBuildState, shard: DataPackShardBuildState) -> DataPackBuildState:
    shards = tuple(
        shard if entry.ordinal == shard.ordinal else entry for entry in state.shards
    )
    completed_shards = sum(1 for updated in shards if updated.status is DataPackShardStatus.READY)
    return replace(state, shards=shards, completed_shards=completed_shards)


def persist_build_state(path: Path, state: DataPackBuildState) -> None:
    write_build_state_file(path, state)


def load_build_state(path: Path) -> DataPackBuildState:
    return read_build_state_file(path)


def discard_shard_temp_outputs(
    *,
    relational_dir: Path,
    embeddings_dir: Path,
    shard_ordinal: int,
) -> None:
    for directory in (relational_dir, embeddings_dir):
        temp_path = temp_shard_path(directory, shard_ordinal)
        if temp_path.exists():
            temp_path.unlink()


def assert_pending_shard_has_no_final_files(
    *,
    relational_dir: Path,
    embeddings_dir: Path,
    shard_ordinal: int,
) -> None:
    for directory in (relational_dir, embeddings_dir):
        final_path = final_shard_path(directory, shard_ordinal)
        if final_path.exists():
            raise VpiDataPackResumeError(
                f"orphan final shard file for PENDING shard {shard_ordinal}: {final_path.name}"
            )


def recover_non_ready_shard(
    shard: DataPackShardBuildState,
    *,
    relational_dir: Path,
    embeddings_dir: Path,
) -> DataPackShardBuildState:
    if shard.status is DataPackShardStatus.READY:
        return shard
    if shard.status is DataPackShardStatus.PENDING:
        assert_pending_shard_has_no_final_files(
            relational_dir=relational_dir,
            embeddings_dir=embeddings_dir,
            shard_ordinal=shard.ordinal,
        )
        return shard
    discard_shard_temp_outputs(
        relational_dir=relational_dir,
        embeddings_dir=embeddings_dir,
        shard_ordinal=shard.ordinal,
    )
    for directory in (relational_dir, embeddings_dir):
        final_path = final_shard_path(directory, shard.ordinal)
        if final_path.exists():
            final_path.unlink()
    return reset_shard_to_pending(shard)


def validate_ready_shard_artifacts(
    *,
    pack_root: Path,
    shard: DataPackShardBuildState,
    relational_schema_version: str,
    embedding_schema_version: str,
    expected_dimension: int,
) -> None:
    if shard.status is not DataPackShardStatus.READY:
        raise VpiDataPackResumeError(f"shard {shard.ordinal} is not READY")
    if shard.relational_relative_path is None or shard.embedding_relative_path is None:
        raise VpiDataPackReadyShardCorruptionError(
            f"READY shard {shard.ordinal} missing relative paths in build state"
        )
    relational_path = pack_root / shard.relational_relative_path
    embedding_path = pack_root / shard.embedding_relative_path
    if not relational_path.is_file() or not embedding_path.is_file():
        raise VpiDataPackReadyShardCorruptionError(
            f"READY shard {shard.ordinal} files missing on disk"
        )
    if shard.relational_sha256 is None or shard.embedding_sha256 is None:
        raise VpiDataPackReadyShardCorruptionError(
            f"READY shard {shard.ordinal} missing sha256 metadata in build state"
        )
    if sha256_file(relational_path) != shard.relational_sha256:
        raise VpiDataPackReadyShardCorruptionError(
            f"READY relational shard {shard.ordinal} sha256 mismatch"
        )
    if sha256_file(embedding_path) != shard.embedding_sha256:
        raise VpiDataPackReadyShardCorruptionError(
            f"READY embedding shard {shard.ordinal} sha256 mismatch"
        )
    relational_descriptor = ShardDescriptor(
        ordinal=shard.ordinal,
        relative_path=shard.relational_relative_path,
        record_count=shard.expected_record_count,
        sha256=shard.relational_sha256,
        source_ref_count=shard.expected_record_count,
        source_ref_set_sha256=shard.relational_source_ref_set_sha256 or "",
        schema_version=relational_schema_version,
    )
    embedding_descriptor = ShardDescriptor(
        ordinal=shard.ordinal,
        relative_path=shard.embedding_relative_path,
        record_count=shard.expected_record_count,
        sha256=shard.embedding_sha256,
        source_ref_count=shard.expected_record_count,
        source_ref_set_sha256=shard.embedding_source_ref_set_sha256 or "",
        schema_version=embedding_schema_version,
    )
    checks = (
        *validate_relational_shard_source_identity(pack_root, relational_descriptor),
        *validate_embedding_shard_source_identity(
            pack_root,
            embedding_descriptor,
            expected_dimension=expected_dimension,
        ),
        *validate_shard_pair_identity(relational_descriptor, embedding_descriptor),
    )
    failures = [check for check in checks if check.status is not ValidationStatus.PASS]
    if failures:
        detail = failures[0].detail
        raise VpiDataPackReadyShardCorruptionError(
            f"READY shard {shard.ordinal} validation failed: {detail}"
        )


def shard_descriptor_paths(shard_ordinal: int) -> tuple[str, str]:
    file_name = shard_file_name(shard_ordinal)
    return f"relational/{file_name}", f"embeddings/{file_name}"
