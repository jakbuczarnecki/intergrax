"""Reusable shard file and source-ref integrity helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.application.domain.source import (
    SourceRecordRef,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    sha256_file,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    source_ref_key,
    source_ref_set_sha256_from_keys,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.relational import (
    RelationalDataPackRecord,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardDescriptor,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.embedding_codec import (
    read_embedding_parquet,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.stores.parquet.relational_codec import (
    read_relational_parquet,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationStatus,
)


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        detail=detail,
    )


def source_ref_keys_from_refs(source_refs: Iterable[SourceRecordRef]) -> tuple[tuple[str, str, str | None], ...]:
    return tuple(source_ref_key(source_ref) for source_ref in source_refs)


def compute_source_ref_set_sha256(source_refs: Iterable[SourceRecordRef]) -> str:
    return source_ref_set_sha256_from_keys(source_ref_keys_from_refs(source_refs))


def validate_no_duplicate_source_refs(
    keys: tuple[tuple[str, str, str | None], ...],
    *,
    check_name: str,
) -> ValidationCheck:
    unique_count = len(set(keys))
    return _check(
        check_name,
        unique_count == len(keys),
        "duplicate source_record_ref detected" if unique_count != len(keys) else "unique",
    )


def validate_global_row_index_ascending(
    records: tuple[RelationalDataPackRecord, ...],
    *,
    check_name: str,
) -> ValidationCheck:
    ascending = all(
        records[index].global_row_index < records[index + 1].global_row_index
        for index in range(len(records) - 1)
    )
    return _check(
        check_name,
        ascending,
        "global_row_index must be strictly ascending within shard",
    )


def validate_shard_descriptor_file(
    pack_root: Path,
    descriptor: ShardDescriptor,
    *,
    check_name_prefix: str,
) -> tuple[ValidationCheck, ...]:
    shard_path = pack_root / descriptor.relative_path
    if not shard_path.is_file():
        return (
            _check(
                f"{check_name_prefix}_missing_{descriptor.ordinal}",
                False,
                descriptor.relative_path,
            ),
        )
    actual = sha256_file(shard_path)
    return (
        _check(
            f"{check_name_prefix}_sha256_{descriptor.ordinal}",
            actual == descriptor.sha256,
            descriptor.relative_path,
        ),
    )


def validate_shard_source_ref_digest(
    *,
    actual_digest: str,
    descriptor: ShardDescriptor,
    check_name_prefix: str,
) -> ValidationCheck:
    return _check(
        f"{check_name_prefix}_source_ref_set_sha256_{descriptor.ordinal}",
        actual_digest == descriptor.source_ref_set_sha256,
        (
            f"descriptor={descriptor.source_ref_set_sha256} "
            f"actual={actual_digest}"
        ),
    )


def validate_relational_shard_source_identity(
    pack_root: Path,
    descriptor: ShardDescriptor,
) -> tuple[ValidationCheck, ...]:
    shard_path = pack_root / descriptor.relative_path
    if not shard_path.is_file():
        return ()
    try:
        records = read_relational_parquet(shard_path)
    except VpiDataPackIntegrityError as exc:
        return (
            _check(
                f"relational_shard_read_{descriptor.ordinal}",
                False,
                str(exc),
            ),
        )
    keys = source_ref_keys_from_refs(record.source_ref for record in records)
    checks = [
        _check(
            f"relational_shard_record_count_{descriptor.ordinal}",
            len(records) == descriptor.record_count,
            f"descriptor={descriptor.record_count} actual={len(records)}",
        ),
        validate_no_duplicate_source_refs(
            keys,
            check_name=f"relational_shard_duplicate_refs_{descriptor.ordinal}",
        ),
    ]
    if len(records) > 1:
        checks.append(
            validate_global_row_index_ascending(
                records,
                check_name=f"relational_shard_row_order_{descriptor.ordinal}",
            )
        )
    checks.append(
        validate_shard_source_ref_digest(
            actual_digest=compute_source_ref_set_sha256(record.source_ref for record in records),
            descriptor=descriptor,
            check_name_prefix="relational_shard",
        )
    )
    return tuple(checks)


def validate_embedding_shard_source_identity(
    pack_root: Path,
    descriptor: ShardDescriptor,
    *,
    expected_dimension: int,
) -> tuple[ValidationCheck, ...]:
    shard_path = pack_root / descriptor.relative_path
    if not shard_path.is_file():
        return ()
    try:
        records = read_embedding_parquet(shard_path, expected_dimension=expected_dimension)
    except VpiDataPackIntegrityError as exc:
        return (
            _check(
                f"embedding_shard_read_{descriptor.ordinal}",
                False,
                str(exc),
            ),
        )
    keys = source_ref_keys_from_refs(record.source_ref for record in records)
    return (
        _check(
            f"embedding_shard_record_count_{descriptor.ordinal}",
            len(records) == descriptor.record_count,
            f"descriptor={descriptor.record_count} actual={len(records)}",
        ),
        validate_no_duplicate_source_refs(
            keys,
            check_name=f"embedding_shard_duplicate_refs_{descriptor.ordinal}",
        ),
        validate_shard_source_ref_digest(
            actual_digest=compute_source_ref_set_sha256(record.source_ref for record in records),
            descriptor=descriptor,
            check_name_prefix="embedding_shard",
        ),
    )


def validate_shard_pair_identity(
    relational: ShardDescriptor,
    embedding: ShardDescriptor,
) -> tuple[ValidationCheck, ...]:
    ordinal = relational.ordinal
    return (
        _check(
            f"shard_pair_ordinal_{ordinal}",
            relational.ordinal == embedding.ordinal,
            f"relational={relational.ordinal} embedding={embedding.ordinal}",
        ),
        _check(
            f"shard_pair_record_count_{ordinal}",
            relational.record_count == embedding.record_count,
            (
                f"relational={relational.record_count} "
                f"embedding={embedding.record_count}"
            ),
        ),
        _check(
            f"shard_pair_source_ref_count_{ordinal}",
            relational.source_ref_count == embedding.source_ref_count,
            (
                f"relational={relational.source_ref_count} "
                f"embedding={embedding.source_ref_count}"
            ),
        ),
        _check(
            f"shard_pair_source_ref_set_sha256_{ordinal}",
            relational.source_ref_set_sha256 == embedding.source_ref_set_sha256,
            (
                f"relational={relational.source_ref_set_sha256} "
                f"embedding={embedding.source_ref_set_sha256}"
            ),
        ),
        _check(
            f"shard_pair_source_ref_count_matches_records_{ordinal}",
            relational.source_ref_count == relational.record_count,
            "relational source_ref_count must equal record_count",
        ),
        _check(
            f"shard_pair_embedding_source_ref_count_matches_records_{ordinal}",
            embedding.source_ref_count == embedding.record_count,
            "embedding source_ref_count must equal record_count",
        ),
    )
